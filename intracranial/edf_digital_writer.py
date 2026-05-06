"""Direct digital EDF/BDF writer for intracranial BIDS conversion.

This module bypasses MNE entirely. It loads raw integer samples (int16
from cmlreaders, no Volts conversion at any point) and writes them via
pyedflib's ``writeSamples(..., digital=True)`` API, which puts the ints
straight on disk with no rescaling. Analyst-side MNE reads the file and
applies ``physical = digital * gain + offset`` then multiplies by the SI
factor implied by the ``physical_dimension`` string ("uV"→1e-6,
"nV"→1e-9), recovering Volts losslessly.

The companion ``resolve_edf_units`` helper picks per-channel pmin/pmax/dim
via this priority cascade:

  1. Real values from the source recording's EDF header (if it is an EDF
     and the values are not the placeholder 0..1 sentinel).
  2. Inference from ``system_1_unit_conversions.csv``: gain_uV =
     1e6 / conversion_to_V, with dim='uV' for ≥1 nV resolution and
     'nV' for finer scales.
  3. Data-derived fallback: gain=1.0 µV/LSB (matches the legacy
     converter default of unit_scale=1e6).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyedflib


# Per-container digital range (EDF int16 vs BDF int24).
_CONTAINER_RANGES = {
    "EDF": (-32768, 32767),
    "BDF": (-8388608, 8388607),
}

# Recognized physical_dimension strings (case-insensitive).
_VALID_DIMENSIONS = {"uv", "µv", "nv", "mv", "v"}


# ----------------------------------------------------------------------
# Source-EDF header inspection
# ----------------------------------------------------------------------

def read_source_edf_units(
    path: str,
) -> Optional[Dict[str, Tuple[float, float, int, int, str]]]:
    """Return ``{label: (pmin, pmax, dmin, dmax, dim)}`` from a source EDF.

    Returns ``None`` if the file does not exist or cannot be opened.
    Returned values are *raw* — placeholder ``pmin=0, pmax=1`` headers
    are returned as-is so the caller can decide whether to honour them
    via :func:`is_placeholder_units`.
    """
    import os

    if not path or not os.path.exists(path):
        return None
    try:
        f = pyedflib.EdfReader(path)
    except Exception:
        return None
    try:
        labels = list(f.getSignalLabels())
        result: Dict[str, Tuple[float, float, int, int, str]] = {}
        for i, label in enumerate(labels):
            result[label] = (
                float(f.getPhysicalMinimum(i)),
                float(f.getPhysicalMaximum(i)),
                int(f.getDigitalMinimum(i)),
                int(f.getDigitalMaximum(i)),
                f.getPhysicalDimension(i).strip(),
            )
        return result
    finally:
        f.close()


def is_placeholder_units(pmin: float, pmax: float, dim: str) -> bool:
    """Return True when an EDF header carries the canonical placeholder.

    The CML system 1 EDFs at ``current_source/raw_eeg/`` all have
    ``pmin=0, pmax=1`` because their real calibration lives in a CML
    sidecar. We treat that pattern (or any unrecognized dimension
    string) as "do not trust the header — fall through to CSV inference".
    """
    if pmin == 0.0 and pmax == 1.0:
        return True
    if dim is None:
        return True
    if dim.strip().lower() not in _VALID_DIMENSIONS:
        return True
    return False


# ----------------------------------------------------------------------
# CSV inference
# ----------------------------------------------------------------------

def infer_units_from_csv(conversion_to_V: float) -> Tuple[float, str]:
    """Return ``(gain, dim)`` for a session given its CSV conversion factor.

    ``conversion_to_V`` is "LSB per Volt" (the column name in
    ``system_1_unit_conversions.csv``). We invert it to gain-per-LSB and
    pick the most natural display dimension:

    - 1 µV/LSB systems        → gain=1.0,  dim='uV'
    - 0.25 µV/LSB systems     → gain=0.25, dim='uV'
    - 1 nV/LSB and finer      → gain in nV, dim='nV'
    """
    if conversion_to_V is None or conversion_to_V <= 0:
        raise ValueError(f"invalid conversion_to_V: {conversion_to_V!r}")
    gain_uv = 1e6 / float(conversion_to_V)
    if gain_uv >= 1e-3:
        return gain_uv, "uV"
    # Sub-nV-resolution: switch to nV so pmin/pmax stay in a nice range.
    gain_nv = gain_uv * 1e3
    return gain_nv, "nV"


def units_for_container(
    gain: float,
    dim: str,
    container: str,
) -> Tuple[float, float, int, int]:
    """Map (gain, dim, container) → (pmin, pmax, dmin, dmax).

    The container determines the digital range; gain * dmin/dmax gives
    the physical range in the units implied by ``dim``.
    """
    if container not in _CONTAINER_RANGES:
        raise ValueError(f"unknown container {container!r}")
    dmin, dmax = _CONTAINER_RANGES[container]
    pmin = float(dmin) * float(gain)
    pmax = float(dmax) * float(gain)
    return pmin, pmax, dmin, dmax


# ----------------------------------------------------------------------
# Priority cascade
# ----------------------------------------------------------------------

def resolve_edf_units(
    channel_labels: Sequence[str],
    *,
    source_edf_path: Optional[str],
    conversion_to_V: Optional[float],
    container: str = "EDF",
    data_for_fallback: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Tuple[float, float, int, int, str]], str]:
    """Return per-channel ``(pmin, pmax, dmin, dmax, dim)`` and a status note.

    Priority cascade:
      1. Source EDF header (per channel — only if not a placeholder).
      2. ``conversion_to_V`` from system_1_unit_conversions.csv.
      3. Data-derived: gain=1.0 µV/LSB and the actual sample range.

    The status string is one of ``"source_edf"``, ``"csv"``, or
    ``"derived"`` and should be propagated to ``df_status`` so a human
    can audit sessions that fell through to derived units.
    """
    src = read_source_edf_units(source_edf_path) if source_edf_path else None

    # ---- Step 1: source EDF header ------------------------------------
    if src is not None:
        usable: Dict[str, Tuple[float, float, int, int, str]] = {}
        for label in channel_labels:
            if label in src:
                pmin, pmax, dmin, dmax, dim = src[label]
                if not is_placeholder_units(pmin, pmax, dim):
                    usable[label] = (pmin, pmax, dmin, dmax, dim)
        if len(usable) == len(channel_labels):
            return usable, "source_edf"

    # ---- Step 2: CSV inference ----------------------------------------
    if conversion_to_V is not None:
        try:
            gain, dim = infer_units_from_csv(conversion_to_V)
            pmin, pmax, dmin, dmax = units_for_container(gain, dim, container)
            return (
                {label: (pmin, pmax, dmin, dmax, dim) for label in channel_labels},
                "csv",
            )
        except ValueError:
            pass  # fall through

    # ---- Step 3: data-derived fallback --------------------------------
    if data_for_fallback is not None:
        observed_min = int(np.min(data_for_fallback))
        observed_max = int(np.max(data_for_fallback))
    else:
        observed_min, observed_max = -32768, 32767
    cmin, cmax = _CONTAINER_RANGES[container]
    dmin = max(cmin, observed_min - 1)
    dmax = min(cmax, observed_max + 1)
    if dmin == dmax:
        # Avoid pmin == pmax which pyedflib rejects.
        dmin -= 1
        dmax += 1
    gain = 1.0  # safe default: 1 µV/LSB
    pmin = float(dmin) * gain
    pmax = float(dmax) * gain
    return (
        {label: (pmin, pmax, dmin, dmax, "uV") for label in channel_labels},
        "derived",
    )


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------

def _container_filetype(container: str) -> int:
    if container == "EDF":
        return pyedflib.FILETYPE_EDFPLUS
    if container == "BDF":
        return pyedflib.FILETYPE_BDFPLUS
    raise ValueError(f"unknown container {container!r}")


def write_digital(
    path: str,
    labels: Sequence[str],
    signals_int: np.ndarray,
    sfreq: float,
    signal_units: Dict[str, Tuple[float, float, int, int, str]],
    *,
    container: str = "EDF",
) -> None:
    """Write integer samples directly to an EDF or BDF file.

    Parameters
    ----------
    path
        Destination file path. The caller is responsible for the
        extension matching ``container`` (".edf" / ".bdf").
    labels
        Per-channel labels in the same order as ``signals_int`` rows.
        Labels longer than 16 chars are silently truncated to fit the
        EDF/BDF spec — the caller should pre-truncate consistently if
        a separate channel-name map needs to be written.
    signals_int
        Integer sample array of shape ``(n_channels, n_samples)``.
        Must be int16 for EDF and int16 or int32 for BDF (BDF stores
        int24 internally; pyedflib accepts int32 input).
    sfreq
        Sampling frequency in Hz.
    signal_units
        Dict from :func:`resolve_edf_units`: maps each label to
        ``(pmin, pmax, dmin, dmax, dim)``.
    container
        ``"EDF"`` (int16, FILETYPE_EDFPLUS) or
        ``"BDF"`` (int24, FILETYPE_BDFPLUS).
    """
    n_channels = len(labels)
    if signals_int.shape[0] != n_channels:
        raise ValueError(
            f"signals_int shape {signals_int.shape} does not match "
            f"len(labels)={n_channels}"
        )
    if container == "EDF":
        if signals_int.dtype != np.int16:
            # pyedflib's digital writer expects int dtype; int16 is the
            # only safe choice for EDF.
            if (signals_int.min() < -32768) or (signals_int.max() > 32767):
                raise ValueError(
                    "samples exceed int16 range — promote container to BDF"
                )
            signals_int = signals_int.astype(np.int16)

    headers: List[dict] = []
    for label in labels:
        if label not in signal_units:
            raise KeyError(f"signal_units missing entry for {label!r}")
        pmin, pmax, dmin, dmax, dim = signal_units[label]
        if pmin == pmax:
            raise ValueError(
                f"{label}: physical_min == physical_max ({pmin}); pyedflib will reject"
            )
        headers.append({
            "label": label[:16],
            "dimension": dim,
            "sample_frequency": float(sfreq),
            "physical_min": float(pmin),
            "physical_max": float(pmax),
            "digital_min": int(dmin),
            "digital_max": int(dmax),
            "transducer": "",
            "prefilter": "",
        })

    writer = pyedflib.EdfWriter(
        str(path),
        n_channels=n_channels,
        file_type=_container_filetype(container),
    )
    try:
        writer.setSignalHeaders(headers)
        # writeSamples expects a list of 1-D arrays per channel (one per
        # signal); pyedflib will accept a 2-D array too in newer versions
        # but the list form works on every version we've shipped against.
        writer.writeSamples(
            [np.ascontiguousarray(signals_int[i]) for i in range(n_channels)],
            digital=True,
        )
    finally:
        writer.close()

"""Read raw int16 samples directly from an EGI simple-binary .raw file.

Used to short-circuit MNE's Volts decode when writing a bit-exact BDF —
the EGI digital int16 values are streamed straight through write_digital,
no requantization. Only supports the int16 precision-2 sub-format; the
caller should fall back to MNE/Volts for float32/64 .raw or .mff inputs.
"""
from __future__ import annotations

import numpy as np


class EGIPrecisionUnsupportedError(ValueError):
    pass


def read_egi_raw_digital(path):
    """Return (data_int16, cal_uV_per_lsb, sfreq, n_eeg_channels).

    data_int16 has shape (n_channels, n_samples) — EEG/data channels
    only, with event/sync channels stripped. Raises
    EGIPrecisionUnsupportedError for float32/64 EGI .raw files.
    """
    with open(path, "rb") as fid:
        version = int(np.fromfile(fid, "<i4", 1)[0])
        precision = version & 6
        if precision != 2:
            raise EGIPrecisionUnsupportedError(
                f"EGI precision={precision} (not int16) at {path}"
            )
        np.fromfile(fid, ">i2", 6)   # year, month, day, hour, minute, second
        np.fromfile(fid, ">i4", 1)   # millisecond
        samp_rate    = int(np.fromfile(fid, ">i2", 1)[0])
        n_channels   = int(np.fromfile(fid, ">i2", 1)[0])
        _gain        = int(np.fromfile(fid, ">i2", 1)[0])
        bits         = int(np.fromfile(fid, ">i2", 1)[0])
        value_range  = int(np.fromfile(fid, ">i2", 1)[0])
        n_samples    = int(np.fromfile(fid, ">i4", 1)[0])
        n_events     = int(np.fromfile(fid, ">i2", 1)[0])
        fid.seek(n_events * 4, 1)  # skip event-code labels
        n_total = n_channels + n_events
        flat = np.fromfile(fid, ">i2", n_total * n_samples)
    if flat.size != n_total * n_samples:
        raise ValueError(
            f"EGI .raw truncated: got {flat.size}, expected {n_total * n_samples}"
        )
    arr = flat.reshape(n_samples, n_total).T
    data = np.ascontiguousarray(arr[:n_channels])
    cal_uV = (value_range / (2.0 ** bits)) if (value_range and bits) else 1.0
    return data, float(cal_uV), float(samp_rate), n_channels

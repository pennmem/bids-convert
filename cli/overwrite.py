"""Resolve ``--overwrite`` tokens into the per-stage overrides dict.

``--overwrite`` is the single overwrite mechanism: bare, it forces every stage
to re-run; with component names, only those stages. Anything not selected keeps
the default resume behavior (a stage runs only when its outputs are missing).

Components are written in a shared vocabulary that fans out to whichever stages
the modality actually has, so ``--overwrite eeg`` means the same thing to a
user regardless of which converter is behind it.
"""

from __future__ import annotations

from .registry import INTRACRANIAL, SCALP, STAGES_BY_MODALITY

ALL_TOKEN = "all"

# Friendly component -> stages, per modality. Checked before the canonical
# stage names, so on the intracranial side `electrodes` deliberately means
# both the monopolar and bipolar electrode stages; name `electrodes`'
# individual stages (`bi-electrodes`) to be more specific.
_ALIASES = {
    SCALP: {
        "beh": ("behavioral",),
        "behavioral": ("behavioral",),
        "events": ("behavioral",),
        "eeg": ("eeg",),
        "montage": ("montage",),
        "electrodes": ("montage",),
        "channels": ("montage",),
    },
    INTRACRANIAL: {
        "beh": ("behavioral",),
        "behavioral": ("behavioral",),
        "events": ("behavioral",),
        "eeg": ("mono-eeg", "bi-eeg"),
        "montage": ("electrodes", "bi-electrodes", "mono-channels", "bi-channels"),
        "electrodes": ("electrodes", "bi-electrodes"),
        "channels": ("mono-channels", "bi-channels"),
        "mono": ("mono-eeg", "mono-channels"),
        "monopolar": ("mono-eeg", "mono-channels"),
        "bi": ("bi-eeg", "bi-channels", "bi-electrodes"),
        "bipolar": ("bi-eeg", "bi-channels", "bi-electrodes"),
    },
}


def valid_tokens(modality: str) -> list[str]:
    """Every token accepted by ``--overwrite`` for this modality."""
    stages = STAGES_BY_MODALITY[modality]
    return sorted({ALL_TOKEN, *_ALIASES[modality], *stages})


def resolve_overwrite(tokens, modality: str) -> dict[str, bool]:
    """Map ``--overwrite`` tokens to ``{stage: bool}``.

    ``tokens is None``  -> flag absent, nothing forced (resume behavior).
    ``tokens == []``    -> bare ``--overwrite``, every stage forced.
    """
    stages = STAGES_BY_MODALITY[modality]
    overrides = {stage: False for stage in stages}

    if tokens is None:
        return overrides
    if not tokens:
        return {stage: True for stage in stages}

    aliases = _ALIASES[modality]
    for token in tokens:
        key = token.strip().lower()
        if key == ALL_TOKEN:
            return {stage: True for stage in stages}
        if key in aliases:
            selected = aliases[key]
        else:
            # Canonical stage names are matched case-insensitively.
            match = [s for s in stages if s.lower() == key]
            if not match:
                raise ValueError(
                    f"--overwrite: unknown component {token!r} for modality "
                    f"{modality!r}. Valid components: {', '.join(valid_tokens(modality))}"
                )
            selected = tuple(match)
        for stage in selected:
            overrides[stage] = True

    return overrides

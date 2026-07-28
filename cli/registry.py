"""Experiment registry — the single source of truth for what an experiment is.

Maps an experiment name to its modality, its converter class, and how that
class is constructed. Converter modules are imported lazily (on first use) so
class-level work in each converter (e.g. loading wordpool files) only happens
for the experiments actually being converted.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
from dataclasses import dataclass

from . import REPO_ROOT

SCALP = "scalp"
INTRACRANIAL = "intracranial"
MODALITIES = (SCALP, INTRACRANIAL)

# Stage vocabularies. These are the canonical names accepted by --overwrite
# and the keys of the `overrides` dict consumed by `_should_run`.
STAGES_BY_MODALITY = {
    SCALP: ("behavioral", "eeg", "montage"),
    INTRACRANIAL: (
        "behavioral", "electrodes", "bi-electrodes",
        "mono-eeg", "bi-eeg", "mono-channels", "bi-channels",
    ),
}

# Subjects excluded by default. The scalp list is the long-standing set of
# test/pilot IDs; intracranial has no equivalent.
DEFAULT_EXCLUDE_SUBJECTS = {
    SCALP: ["LTP001", "LTP9000", "LTP9001"],
    INTRACRANIAL: [],
}

_INTRACRANIAL_DIR = os.path.join(REPO_ROOT, "intracranial")


@dataclass(frozen=True)
class ExperimentSpec:
    """How to build the converter for one experiment.

    ``module_path`` is a dotted module name, or an absolute path to a .py file
    for modules whose directory name is not a Python identifier (``PS2.1``).
    ``ctor`` selects the constructor calling convention.
    """

    name: str
    modality: str
    module_path: str
    class_name: str
    ctor: str = INTRACRANIAL  # "scalp" | "intracranial" | "pyFR"

    def load_class(self):
        if self.module_path.endswith(".py"):
            spec = importlib.util.spec_from_file_location(self.class_name, self.module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        else:
            module = importlib.import_module(self.module_path)
        return getattr(module, self.class_name)

    def build(self, subject, session, *, root, overrides, job=None):
        """Construct (but do not run) the converter for one session.

        ``job`` carries the intracranial-only per-session parameters
        (system_version, unit_scale, brain_regions); scalp ignores it.
        Construction is deliberately cheap and I/O-free for every modality so
        the orchestrator can ask the converter which stages it would run
        before committing to any work.
        """
        cls = self.load_class()
        job = job or {}

        if self.ctor == SCALP:
            return cls(subject, self.name, session, root=root, overrides=overrides)

        if self.ctor == "pyFR":
            # pyFR predates the common signature: (subject, experiment,
            # session, montage, math_events, system_version, unit_scale,
            # monopolar, bipolar, mni, tal, overrides=, root=).
            return cls(
                subject, self.name, session,
                0,      # montage
                False,  # math_events
                job["system_version"], job["unit_scale"],
                True,   # monopolar
                True,   # bipolar
                True,   # mni
                False,  # tal
                overrides=overrides, root=root,
            )

        return cls(
            subject, self.name, session,
            job["system_version"], job["unit_scale"],
            False,  # area
            job.get("brain_regions"),
            overrides=overrides, root=root,
        )


def _intracranial(name, package=None, class_name=None, ctor=INTRACRANIAL):
    package = package or name
    class_name = class_name or f"{name}_BIDS_converter"
    return ExperimentSpec(
        name=name,
        modality=INTRACRANIAL,
        module_path=f"intracranial.{package}.{package}_BIDS_converter",
        class_name=class_name,
        ctor=ctor,
    )


def _scalp(name):
    return ExperimentSpec(
        name=name,
        modality=SCALP,
        module_path="scalp.ScalpBIDSConverter",
        class_name="ScalpBIDSConverter",
        ctor=SCALP,
    )


_SPECS = [
    # ---- intracranial ----
    _intracranial("catFR1"),
    _intracranial("catFR2"),
    _intracranial("FR1"),
    _intracranial("FR2"),
    _intracranial("ICatFR1"),
    _intracranial("IFR1"),
    _intracranial("PAL1"),
    _intracranial("PAL2"),
    _intracranial("pyFR", ctor="pyFR"),
    _intracranial("RepFR1"),
    _intracranial("YC1"),
    _intracranial("YC2"),
    _intracranial("PS2"),
    # The PS2.1 folder name contains a dot, so it cannot be imported via
    # importlib.import_module — load it from its absolute path instead.
    ExperimentSpec(
        name="PS2.1",
        modality=INTRACRANIAL,
        module_path=os.path.join(_INTRACRANIAL_DIR, "PS2.1", "PS2.1_BIDS_converter.py"),
        class_name="PS21_BIDS_converter",
    ),
    # ---- scalp ----
    # One flat converter handles all of these; the set is bounded by the
    # experiments ScalpBIDSConverter.event_column_dict knows how to describe.
    _scalp("ValueCourier"),
    _scalp("ltpFR"),
    _scalp("ltpFR2"),
    _scalp("VFFR"),
    _scalp("VCBehOnly"),
    _scalp("VCFROP"),
    _scalp("CourierReinstate1"),
]

EXPERIMENTS: dict[str, ExperimentSpec] = {spec.name: spec for spec in _SPECS}


def get(experiment: str) -> ExperimentSpec:
    try:
        return EXPERIMENTS[experiment]
    except KeyError:
        raise ValueError(
            f"Unknown experiment {experiment!r}. Known experiments: "
            f"{', '.join(sorted(EXPERIMENTS))}"
        ) from None


def experiments_for(modality: str) -> list[str]:
    return sorted(name for name, spec in EXPERIMENTS.items() if spec.modality == modality)


def resolve_modality(experiments, explicit=None) -> str:
    """Determine the modality for this run.

    Inferred from ``--experiments`` when possible; ``--modality`` is required
    when no experiments were named, and wins if both are given (it is then
    checked for consistency).
    """
    if experiments:
        modalities = {get(exp).modality for exp in experiments}
        if len(modalities) > 1:
            raise ValueError(
                "--experiments mixes modalities "
                f"({', '.join(sorted(modalities))}); run one modality at a time."
            )
        inferred = modalities.pop()
        if explicit and explicit != inferred:
            raise ValueError(
                f"--modality {explicit} conflicts with the selected experiments, "
                f"which are {inferred}."
            )
        return inferred

    if explicit:
        return explicit

    raise ValueError(
        "Cannot determine modality: pass --modality {scalp,intracranial} "
        "or name experiments with --experiments."
    )

# bids-convert

Tools for converting CML (Computational Memory Lab) data to the [Brain Imaging Data Structure (BIDS)](https://bids-specification.readthedocs.io/en/stable/) format, suitable for upload to [OpenNeuro](https://openneuro.org).

**Dependencies:** [CMLReaders](https://github.com/pennmem/cmlreaders), [MNE-BIDS](https://github.com/mne-tools/mne-bids)

---

## Repository structure

```
bids-convert/
├── bids_convert.py             # THE entry point — scalp and intracranial
├── cli/                        # shared conversion engine (used by both modalities)
│   ├── registry.py             # experiment -> modality + converter class
│   ├── stages.py               # stage gating, failure policy, root BIDS files
│   ├── overwrite.py            # --overwrite components -> per-stage overrides
│   ├── jobs.py                 # job table from the CML data index
│   ├── runner.py               # serial + Slurm/Dask orchestration, error logging
│   └── validation.py           # post-conversion validation
├── bids_validation.py          # BIDS Validator + eeg-validation pipelines
├── conversion_error_log.py     # per-task conversion error CSV
├── intracranial/               # iEEG-BIDS converters
│   ├── intracranial_BIDS_converter.py   # base class (all iEEG converters inherit this)
│   ├── intracranial_BIDS_metadata.py    # pre-conversion metadata checker
│   ├── run_BIDS_metadata.py             # CLI wrapper for metadata checker
│   ├── edf_digital_writer.py            # digital EDF/BDF writer (shared with scalp)
│   ├── system_1_unit_conversions.csv    # unit scale per session for system-1 recordings
│   ├── system_versions.csv              # resolved system versions for sessions with NaN in data index
│   ├── bids_brain_regions.csv           # number of contacts with valid region labels per session
│   ├── FR1/                 # Free Recall 1
│   ├── FR2/                 # Free Recall 2
│   ├── catFR1/              # Categorized Free Recall 1
│   ├── catFR2/              # Categorized Free Recall 2
│   ├── IFR1/                # Free Recall 1, Elemem/System-4 implementation
│   ├── ICatFR1/             # Categorized Free Recall 1, Elemem/System-4 implementation
│   ├── RepFR1/              # Repeated Free Recall 1
│   ├── PAL1/                # Paired Associates Learning 1
│   ├── PAL2/                # Paired Associates Learning 2
│   ├── pyFR/                # pyFR (system-1 era free recall)
│   ├── YC1/                 # Yellow Cab spatial navigation 1
│   ├── YC2/                 # Yellow Cab spatial navigation 2
│   ├── PS2/                 # Pulse stimulation 2 (brain stimulation, task-free)
│   └── PS2.1/               # Pulse stimulation 2.1
└── scalp/                      # scalp EEG converters (ltpFR, ltpFR2, VFFR, ValueCourier, ...)
    ├── ScalpBIDSConverter.py           # the scalp converter
    ├── run_scalp_converter.sh          # maint/cron wrapper (recently-modified sessions)
    └── convert.py                      # single-session helper
```

Each intracranial experiment folder contains a single `<Experiment>_BIDS_converter.py` with a class that inherits from `intracranial_BIDS_converter` and overrides the experiment-specific methods (`set_wordpool`, `events_to_BIDS`, `apply_event_durations`, `make_events_descriptor`, `eeg_sidecar`). Scalp has one flat converter class covering every scalp experiment.

---

## Running a conversion

`bids_convert.py` is the entry point for the whole repo. The modality is inferred
from the experiments you name (`--modality` disambiguates when you name none):

```bash
cd ~/bids-convert

# One intracranial subject / experiment / session, one at a time
python bids_convert.py --experiments FR1 --subjects R1001P --sessions 0 --serial \
    --root /path/to/BIDS

# Sessions 0-4 for two subjects, in parallel over Slurm+Dask (the default)
python bids_convert.py --experiments FR1 --subjects R1001P R1002P --sessions 0:5 \
    --root /path/to/BIDS

# A whole scalp experiment
python bids_convert.py --experiments ltpFR2 --root /data/LTP_BIDS/ltpFR2

# See what would run, without converting anything
python bids_convert.py --modality scalp --smokescreen --root /tmp/x --dry-run
```

`--root` is the BIDS dataset root and is used **verbatim** — the experiment name is
never appended. Intracranial experiments conventionally share one dataset
(distinguished by the `task-` entity); scalp experiments conventionally get one
dataset each (`/data/LTP_BIDS/<Experiment>/`), so pass the full path.

### Selection

| Flag | Default | Description |
|------|---------|-------------|
| `--modality {scalp,intracranial}` | inferred | Required only when no experiments are named |
| `--root` | *(required)* | BIDS output directory, used verbatim |
| `--experiments` | all of the modality | Experiments to convert |
| `--subjects` | all | Subject IDs |
| `--sessions` | all | Ints and/or slices: `3`, `0:5`, `:3`, `2:` (requires `--subjects` or `--experiments`) |
| `--exclude-subjects` | modality test IDs | Subjects to skip |
| `--recently-modified` | — | `recently_modified.json` (`{subject: [sessions]}`) to restrict jobs |
| `--smokescreen` | off | Quick test: 1 subject per experiment |

### Overwriting existing output

By default each stage runs only when its outputs are missing, so re-running a
conversion resumes rather than redoing work. `--overwrite` forces stages to
re-run — bare it means everything, or name components:

```bash
python bids_convert.py --experiments FR1 --overwrite            # every stage
python bids_convert.py --experiments FR1 --overwrite beh        # behavioral only
python bids_convert.py --experiments FR1 --overwrite eeg        # mono-eeg + bi-eeg
python bids_convert.py --experiments ltpFR2 --overwrite eeg     # the scalp eeg stage
python bids_convert.py --experiments FR1 --overwrite bi-eeg electrodes
```

| Component | scalp stages | intracranial stages |
|-----------|--------------|---------------------|
| `all` | everything | everything |
| `beh` / `behavioral` / `events` | behavioral | behavioral |
| `eeg` | eeg | mono-eeg, bi-eeg |
| `montage` | montage | electrodes, bi-electrodes, mono-channels, bi-channels |
| `electrodes` | montage | electrodes, bi-electrodes |
| `channels` | montage | mono-channels, bi-channels |
| `mono` / `monopolar` | — | mono-eeg, mono-channels |
| `bi` / `bipolar` | — | bi-eeg, bi-channels, bi-electrodes |

Exact stage names are also accepted, for finer control than the aliases give.

### Behavior, validation and parallelism

| Flag | Default | Description |
|------|---------|-------------|
| `--serial` | off (parallel) | Run jobs one at a time instead of over Slurm+Dask |
| `--force` | off | Downgrade stage failures to `[WARN]` and keep going; by default a stage failure aborts that session |
| `--dry-run` | off | Print resolved settings + job table, then exit |
| `--verbose` | off | Verbose validation-pipeline output |
| `--validate` | off | Run both validation layers after conversion |
| `--bids-validator` / `--eeg-validator` | off | Run only that layer |
| `--validate-only` | off | Skip conversion, validate `--root` for the selected jobs |
| `--job-name`, `--memory-per-job`, `--max-n-jobs`, `--threads-per-job`, `--adapt`/`--no-adapt`, `--log-directory` | `bids_convert`, `100GB`, `20`, `1`, adapt on, `~/logs/` | Slurm/Dask cluster tuning |
| `--conversion-csv` | `intracranial/system_1_unit_conversions.csv` | Intracranial only: per-session unit conversions |

The process exits non-zero if any session failed or validation did not pass.

### Logs and error reporting

* Per-session conversion stdout/stderr: `/data/BIDS-convert-logs/<experiment>/<subject>/<session>/`
* Per-task failure table: `<root>/bids_conversion_error_<experiment>.csv` (added to `.bidsignore`)

That log root is owned by `RAM_maint`. To run a conversion under your own
account, point it somewhere writable:

```bash
export BIDS_CONVERT_LOG_ROOT=/scratch/$USER/bids_convert_logs
```

A session is only recorded in the error CSV when it actually ran, so a
`skip existing` re-run leaves any prior error rows intact; a session that
succeeds on a later run has its old row removed.

### Scalp maint / cron wrapper

`scalp/run_scalp_converter.sh` reads `/data/eeg/scalp/ltp/ACTIVE_EXPERIMENTS.txt`
and, for each experiment with a `recently_modified.json`, converts all of its
recently-modified sessions in one Slurm+Dask cluster, writing to
`/data/LTP_BIDS/<Experiment>/`. Extra flags are passed straight through:

```bash
bash scalp/run_scalp_converter.sh            # normal maint run
bash scalp/run_scalp_converter.sh --dry-run  # show the jobs it would run
```

---

## iEEG conversion details

### Architecture

```
StageGatedConverter               (cli/stages.py — shared stage gating)
        │
        ├── intracranial_BIDS_converter   (base class)
        │           │
        │           └── <Experiment>_BIDS_converter   (one per experiment folder)
        │
        └── ScalpBIDSConverter            (all scalp experiments)
```

The intracranial base class handles all EEG/electrode/channel I/O (loading via
CMLReaders, writing EDF + BIDS sidecars, electrode coordinates, channel tables).
Subclasses only implement the experiment-specific event logic.

Construction is cheap for both converters — `run()` does the work — so the
orchestrator can ask a converter which stages it would run before committing to
any I/O, and skip a fully-converted session outright.

`intracranial_BIDS_converter.run()` executes:

1. Load events → `events_to_BIDS()` → write `_beh.tsv` + sidecar JSON
2. Load EEG metadata (sample rate, recording duration)
3. Load contacts → write `_electrodes.tsv` + coordinate system JSON
4. Load pairs → write `_channels.tsv`
5. Write `_ieeg.edf` + sidecar JSON (bipolar and/or monopolar)

### System versions and unit scales

| System | Recording units | `unit_scale` to convert to V |
|--------|----------------|-------------------------------|
| 1      | varies          | from `system_1_unit_conversions.csv` |
| 2      | 250 nV          | 4,000,000 |
| 3      | 0.1 μV          | 10,000,000 |
| 4      | 250 nV          | 4,000,000 |

Sessions with `NaN` in the data index system version column are resolved via `system_versions.csv`.

> **PS2 note:** PS2 sessions run on system 2 (`unit_scale=4000000.0`) and are not present in `system_1_unit_conversions.csv`. Sessions missing from the CSV fall back to `unit_scale=1e6`, so add PS2 rows to the CSV or instantiate the converter directly (below).

### Calling a converter directly (Python)

```python
import sys, os
sys.path.insert(0, os.path.expanduser("~/bids-convert"))

from intracranial.PS2.PS2_BIDS_converter import PS2_BIDS_converter

converter = PS2_BIDS_converter(
    subject="R1050M",
    experiment="PS2",
    session=0,
    system_version=2.0,
    unit_scale=4_000_000.0,
    area=False,
    brain_regions={"wb.region": 1, "ind.region": 1, "das.region": 1, "stein.region": 1},
    overrides={},          # {stage: True} to force a stage to re-run
    root="/path/to/BIDS/output/",
)
converter.force = False    # True to warn-and-continue past stage failures
converter.run()
```

---

## BIDS validation

After conversion, validate the output with the [BIDS Validator](https://hub.docker.com/r/bids/validator). The entry point can run this automatically via `--validate` (see options above).

**Docker (recommended on a cluster):**
```bash
docker run --rm -v /path/to/BIDS:/data:ro bids/validator /data
```

**npm (local install):**
```bash
npm install bids-validator
bids-validator /path/to/BIDS
```

**npx (no global install needed):**
```bash
npx bids-validator /path/to/BIDS
```

See [hub.docker.com/r/bids/validator](https://hub.docker.com/r/bids/validator) for full installation and usage instructions.

---

## Running the metadata checker (optional pre-step)

Before converting a new experiment for the first time, run the metadata checker to audit which sessions have loadable events, contacts, pairs, and EEG, and to determine system versions and unit scales:

```bash
cd ~/bids-convert/intracranial
python run_BIDS_metadata.py FR1
```

This writes a `metadata_df.csv` to `<root>/<experiment>/metadata/`. That CSV can then be used to populate `system_1_unit_conversions.csv` for system-1 sessions.

---

## Adding a new experiment

1. Create `intracranial/<NewExp>/` with `<NewExp>_BIDS_converter.py` containing a class that inherits from `intracranial_BIDS_converter` and implements:
   - `set_wordpool()` — return wordpool filename or `'n/a'`
   - `events_to_BIDS()` — load and format events DataFrame
   - `apply_event_durations()` — assign per-event-type durations
   - `make_events_descriptor()` — return BIDS events sidecar dict
   - `eeg_sidecar()` — (optional) override to add `TaskDescription`

2. Register it in `cli/registry.py`:
   ```python
   _SPECS = [
       ...,
       _intracranial("NewExp"),      # or _scalp("NewExp") for a scalp experiment
   ]
   ```
   `_intracranial(name)` assumes `intracranial/<name>/<name>_BIDS_converter.py` defining
   `<name>_BIDS_converter`; pass `package=` / `class_name=` when they differ.

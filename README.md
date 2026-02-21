# bids-convert

Tools for converting CML (Computational Memory Lab) data to the [Brain Imaging Data Structure (BIDS)](https://bids-specification.readthedocs.io/en/stable/) format, suitable for upload to [OpenNeuro](https://openneuro.org).

**Dependencies:** [CMLReaders](https://github.com/pennmem/cmlreaders), [MNE-BIDS](https://github.com/mne-tools/mne-bids)

---

## Repository structure

```
bids-convert/
├── intracranial/               # iEEG-BIDS converters
│   ├── intracranial_BIDS_converter.py   # base class (all iEEG converters inherit this)
│   ├── intracranial_BIDS_metadata.py    # pre-conversion metadata checker
│   ├── run_intracranial_converter.py    # main CLI runner (single / serial / parallel)
│   ├── run_BIDS_metadata.py             # CLI wrapper for metadata checker
│   ├── system_1_unit_conversions.csv    # unit scale per session for system-1 recordings
│   ├── system_versions.csv              # resolved system versions for sessions with NaN in data index
│   ├── bids_brain_regions.csv           # number of contacts with valid region labels per session
│   ├── FR1/                 # Free Recall 1
│   ├── FR2/                 # Free Recall 2
│   ├── catFR1/              # Categorized Free Recall 1
│   ├── catFR2/              # Categorized Free Recall 2
│   ├── RepFR1/              # Repeated Free Recall 1
│   ├── PAL1/                # Paired Associates Learning 1
│   ├── PAL2/                # Paired Associates Learning 2
│   ├── pyFR/                # pyFR (system-1 era free recall)
│   ├── YC1/                 # Yellow Cab spatial navigation 1
│   ├── YC2/                 # Yellow Cab spatial navigation 2
│   ├── PS2/                 # Pulse stimulation 2 (brain stimulation, task-free)
│   └── PS2.1/               # Pulse stimulation 2.1
└── scalp/                      # Scalp EEG converters (PEERS: ltpFR, ltpFR2, VFFR, ValueCourier)
```

Each experiment folder contains a single `<Experiment>_BIDS_converter.py` file with a class that inherits from `intracranial_BIDS_converter` and overrides the experiment-specific methods (`events_to_BIDS`, `apply_event_durations`, `make_events_descriptor`, `eeg_sidecar`).

---

## iEEG conversion

### Architecture

```
intracranial_BIDS_converter   (base class)
        │
        └── <Experiment>_BIDS_converter   (one per experiment folder)
```

The base class handles all EEG/electrode/channel I/O (loading via CMLReaders, writing EDF + BIDS sidecars, electrode coordinates, channel tables). Subclasses only implement the experiment-specific event logic.

The `run()` method on the converter executes the full pipeline in order:

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

---

## Running the iEEG converter

The main entry point is `intracranial/run_intracranial_converter.py`. Run it as a module from the repo root (so relative imports resolve correctly):

```bash
cd ~/bids-convert
python -m intracranial.run_intracranial_converter --mode <MODE> [options]
```

### Modes

#### Single session

```bash
python -m intracranial.run_intracranial_converter \
    --mode single \
    --subject R1001P \
    --experiment FR1 \
    --session 0 \
    --root /path/to/BIDS/output/
```

The conversion params (`system_version`, `unit_scale`) are looked up automatically from `system_1_unit_conversions.csv`. If no matching row exists the session is skipped with a warning.

#### Serial (multiple sessions, one at a time)

```bash
python -m intracranial.run_intracranial_converter \
    --mode serial \
    --experiments FR1 FR2 catFR1 \
    --max-subjects 10 \
    --root /path/to/BIDS/output/
```

#### Parallel (Dask/SLURM)

```bash
python -m intracranial.run_intracranial_converter \
    --mode parallel \
    --experiments FR1 FR2 \
    --max-n-jobs 20 \
    --memory-per-job 50GB \
    --root /path/to/BIDS/output/
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--root` | `/home1/maint/LTP_BIDS/` | BIDS output directory |
| `--conversion-csv` | `system_1_unit_conversions.csv` | Unit conversion table |
| `--experiments` | all supported | Experiments to include (serial/parallel) |
| `--max-subjects` | 10 | Max subjects per experiment (serial/parallel) |
| `--exclude-subjects` | `LTP001` | Subjects to skip |
| `--monopolar` / `--no-monopolar` | on | Write monopolar EEG |
| `--bipolar` / `--no-bipolar` | on | Write bipolar EEG |
| `--mni` / `--no-mni` | on | Write MNI electrode coordinates |
| `--tal` | off | Write Talairach electrode coordinates |
| `--area` | off | Include electrode surface area |

### Supported experiments

| Experiment | Type | Converter class |
|------------|------|-----------------|
| FR1, FR2 | Free recall | `FR1_BIDS_converter`, `FR2_BIDS_converter` |
| catFR1, catFR2 | Categorized free recall | `catFR1_BIDS_converter`, `catFR2_BIDS_converter` |
| RepFR1 | Repeated free recall | `RepFR1_BIDS_converter` |
| PAL1, PAL2 | Paired associates learning | `PAL1_BIDS_converter`, `PAL2_BIDS_converter` |
| pyFR | Free recall (system 1) | `pyFR_BIDS_converter` |
| YC1, YC2 | Spatial navigation | `YC1_BIDS_converter`, `YC2_BIDS_converter` |
| PS2 | Brain stimulation (task-free) | `PS2_BIDS_converter` |

> **PS2 note:** PS2 sessions run on system 2 (`unit_scale=4000000.0`) and are not present in `system_1_unit_conversions.csv`. To convert PS2 data, instantiate the converter directly (see below) or add PS2 rows to the CSV.

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
    monopolar=True,
    bipolar=True,
    mni=True,
    tal=False,
    area=False,
    brain_regions={"wb.region": 1, "ind.region": 1, "das.region": 1, "stein.region": 1},
    root="/path/to/BIDS/output/",
)
converter.run()
```

---

## BIDS validation

After conversion, validate the output with the [BIDS Validator](https://hub.docker.com/r/bids/validator). The converter can run this automatically via `--validate` (see runner options above).

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

2. Register it in `run_intracranial_converter.py`:
   ```python
   from intracranial.NewExp.NewExp_BIDS_converter import NewExp_BIDS_converter
   # ...
   EXPERIMENT_TO_CONVERTER = {
       ...,
       "NewExp": NewExp_BIDS_converter,
   }
   ```
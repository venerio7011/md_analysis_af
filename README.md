# AlphaFold3 to HADDOCK Molecular Dynamics Pipeline

A comprehensive pipeline for molecular dynamics analysis of AlphaFold3 protein-protein interaction predictions using HADDOCK and OpenMM with CHARMM force fields.

## Overview

This pipeline processes AlphaFold3 (AF3) structural predictions and performs:
1. Structure preprocessing and validation
2. HADDOCK docking and refinement
3. OpenMM energy scoring with CHARMM36 force field
4. Trajectory visualization and analysis

## Pipeline Components

### Core Pipeline Modules

| Module | Description |
|--------|-------------|
| `structure_preprocessor.py` | AF3 model discovery, CIF to PDB conversion, confidence extraction |
| `af3_air_pipeline_validator.py` | Validate AF3 predictions for HADDOCK AIR generation |
| `af3_air_validator_v2.py` | Enhanced AF3 AIR validation (v2) |
| `haddock_preparation_validator.py` | Validate structures before HADDOCK submission |
| `haddock_runner.py` | HADDOCK3 execution with restart support |
| `haddock_emscoring.py` | HADDOCK energy minimization scoring |
| `openmm_charmm_scoring.py` | OpenMM + CHARMM36 energy evaluation and minimization |
| `input_zip_manager.py` | Manage input file archives |
| `evidence_cutoff.py` | Apply confidence cutoffs to AF3 evidence |
| `AF3_MD_pipeline_kneedle.ipynb` | Main pipeline notebook with Kneedle analysis |

### Visualization Tools

| Module | Description |
|--------|-------------|
| `trajectory_viewer.py` | DCD trajectory visualization utility |
| `DCD_trajectory_viewer_kneedle.ipynb` | Interactive trajectory analysis with Kneedle method |

## Requirements

```
python >= 3.8
openmm
pdbfixer
gemmi
haddock3
numpy
pandas
matplotlib
mdtraj
nglview
kneed
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/af3-haddock-md-pipeline.git
cd af3-haddock-md-pipeline

# Install dependencies
pip install -r requirements.txt

# Install HADDOCK3 (follow official instructions)
# https://www.bonvinlab.org/haddock3/

# Install OpenMM
conda install -c conda-forge openmm
```

## Usage

### 1. Structure Preprocessing

```python
from structure_preprocessor import process_af3_folder

# Process AF3 output folder
results = process_af3_folder(
    af3_folder="path/to/af3_output",
    output_dir="path/to/processed"
)
```

### 2. HADDOCK Docking

```python
from haddock_runner import run_haddock

# Run HADDOCK3 workflow
result = run_haddock(
    input_pdb="structure.pdb",
    output_dir="haddock_output",
    air_mode="AF3_DERIVED",
    ncores=8
)
```

### 3. OpenMM Energy Scoring

```python
from openmm_charmm_scoring import run_openmm_charmm_batch

# Score structures with CHARMM36
results = run_openmm_charmm_batch(
    candidates=candidate_list,
    results_dir="scoring_output",
    minimize=True,
    md_steps=1000
)
```

### 4. Trajectory Visualization

```python
from trajectory_viewer import view_trajectory

# Visualize MD trajectory
view_trajectory(
    dcd_file="trajectory.dcd",
    topology="structure.pdb"
)
```

### Using the Notebook

Open `AF3_MD_pipeline_kneedle.ipynb` in Jupyter for the complete interactive workflow.

## Pipeline Workflow

```
AlphaFold3 Output (mmCIF)
         │
         ▼
┌─────────────────────┐
│ Structure Processor │  → PDB conversion, validation
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   AIR Generation    │  → Extract interface residues
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   HADDOCK Docking   │  → Rigid body + flexible refinement
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  OpenMM Scoring     │  → CHARMM36 energy minimization
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│ Trajectory Analysis │  → Kneedle clustering, visualization
└─────────────────────┘
```

## Output Files

- `*_minimized.pdb` - Energy minimized structures
- `*.dcd` - MD trajectory files
- `*_results.csv` - Scoring results
- `*.json` - Per-sample metadata

## References

- AlphaFold3: https://alphafold.ebi.ac.uk/
- HADDOCK: https://www.bonvinlab.org/software/haddock2.4/
- OpenMM: https://openmm.org/
- CHARMM Force Field: https://www.charmm.org/

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

venerio7011

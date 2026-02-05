"""Constants for GuacaMol molecular optimization."""

from pathlib import Path

# GuacaMol benchmark tasks (12 total)
GUACAMOL_TASKS = [
    "med1",  # Median molecules 1 (camphor-menthol)
    "pdop",  # Perindopril MPO
    "adip",  # Amlodipine MPO
    "rano",  # Ranolazine MPO
    "osmb",  # Osimertinib MPO
    "siga",  # Sitagliptin MPO
    "zale",  # Zaleplon MPO
    "valt",  # Valsartan SMARTS
    "med2",  # Median molecules 2 (tadalafil-sildenafil)
    "dhop",  # Decoration Hop
    "shop",  # Scaffold Hop
    "fexo",  # Fexofenadine MPO
]

# Additional tasks from NFBO
EXTRA_TASKS = [
    "logp",  # Penalized logP
    "qed",   # Drug-likeness (QED)
]

GUACAMOL_TASK_DESCRIPTIONS = {
    "med1": "Median molecules 1: similarity to camphor and menthol",
    "pdop": "Perindopril MPO: ACE inhibitor optimization",
    "adip": "Amlodipine MPO: calcium channel blocker optimization",
    "rano": "Ranolazine MPO: antianginal optimization",
    "osmb": "Osimertinib MPO: EGFR inhibitor optimization",
    "siga": "Sitagliptin MPO: DPP-4 inhibitor optimization",
    "zale": "Zaleplon MPO: sedative-hypnotic optimization",
    "valt": "Valsartan SMARTS: ARB optimization with SMARTS constraint",
    "med2": "Median molecules 2: similarity to tadalafil and sildenafil",
    "dhop": "Decoration Hop: modify decorations while keeping scaffold",
    "shop": "Scaffold Hop: modify scaffold while keeping activity",
    "fexo": "Fexofenadine MPO: antihistamine optimization (less greasy)",
    "logp": "Penalized logP: lipophilicity with SA and ring penalty",
    "qed": "Quantitative Estimate of Drug-likeness",
}

# Default paths
DEFAULT_DATA_PATH = Path("datasets/guacamol/guacamol_train_data_first_20k.csv")

# Column mapping in CSV
CSV_COLUMNS = {
    "selfies": "selfie",
    "smiles": "smile",
}

# All task columns in the CSV file
CSV_TASK_COLUMNS = GUACAMOL_TASKS + ["logp"]

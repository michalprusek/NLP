"""
Convert tos_converted dataset to tos_local format expected by claudette evaluators
"""
import json
import pandas as pd
from pathlib import Path
import random

def convert_tos_dataset():
    """Convert tos_converted CSV/JSON to tos_local JSON format."""
    
    # Load the converted dataset
    input_file = Path("datasets/tos_converted/tos_dataset.json")
    print(f"Loading dataset from {input_file}...")
    
    df = pd.read_json(input_file)
    print(f"Loaded {len(df)} examples")
    
    # Map column names from tos_converted to tos_local format
    # tos_converted uses: Arbitration, Choice of Law, Content Removal, Jurisdiction, Law, 
    #                     Limitation of Liability, Termination, Unilateral Change
    # tos_local uses: a, law, cr, j, (law), ltd, ter, ch (+ pinc, use)
    
    all_data = []
    for idx, row in df.iterrows():
        example = {
            'sentence': row['text'],
            'company': row['company'],
            'line_number': idx,
            'language': 'en',
            'unfairness_level': 'unfair' if row['is_unfair'] == 1 else 'fair',
            # Map categories (tos_converted → tos_local boolean fields)
            'ltd': bool(row['Limitation of Liability']),  # 0: Limitation of liability
            'ter': bool(row['Termination']),               # 1: Unilateral termination
            'ch': bool(row['Unilateral Change']),          # 2: Unilateral change
            'a': bool(row['Arbitration']),                 # 3: Arbitration
            'cr': bool(row['Content Removal']),            # 4: Content removal
            'law': bool(row['Choice of Law']),             # 5: Choice of law
            'pinc': False,                                  # 6: Other (not in dataset)
            'use': False,                                   # 7: Contract by using (not in tos_converted)
            'j': bool(row['Jurisdiction']),                # 8: Jurisdiction
        }
        all_data.append(example)
    
    print(f"Converted {len(all_data)} examples")
    
    # Split dataset: 70% train, 15% validation, 15% test
    # Shuffle with fixed seed for reproducibility
    random.seed(42)
    random.shuffle(all_data)
    
    n = len(all_data)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    
    splits = {
        'train': all_data[:train_size],
        'validation': all_data[train_size:train_size + val_size],
        'test': all_data[train_size + val_size:],
    }
    
    # Save to tos_local directory
    output_dir = Path("datasets/tos_local")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = output_dir / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {split_name}: {len(split_data)} examples → {output_file}")
    
    # Create metadata
    metadata = {
        "name": "ToS Local Dataset",
        "source": "datasets/tos_converted",
        "description": "Terms of Service fairness classification converted from tos_converted",
        "labels": {
            "0": "Limitation of liability (LTD)",
            "1": "Unilateral termination (TER)",
            "2": "Unilateral change (CH)",
            "3": "Arbitration (A)",
            "4": "Content removal (CR)",
            "5": "Choice of law (LAW)",
            "6": "Other (PINC)",
            "7": "Contract by using (USE)",
            "8": "Jurisdiction (J)"
        },
        "splits": {k: len(v) for k, v in splits.items()},
        "total_examples": len(all_data),
        "num_companies": df['company'].nunique(),
        "unfair_percentage": (df['is_unfair'].sum() / len(df) * 100),
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Dataset converted to {output_dir}!")
    print(f"  Train: {len(splits['train'])} examples")
    print(f"  Validation: {len(splits['validation'])} examples")
    print(f"  Test: {len(splits['test'])} examples")
    print(f"  Companies: {metadata['num_companies']}")
    print(f"  Unfair: {metadata['unfair_percentage']:.2f}%")

if __name__ == "__main__":
    print("="*80)
    print("Converting ToS Dataset: tos_converted → tos_local")
    print("="*80)
    convert_tos_dataset()


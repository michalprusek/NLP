#!/usr/bin/env python3
"""
Visualization script for ToS dataset statistics.
"""
import pandas as pd
import json
from pathlib import Path

def print_summary():
    """Print dataset summary with ASCII visualization."""

    # Load statistics
    stats_path = Path(__file__).parent / "dataset_statistics.json"
    with open(stats_path, 'r') as f:
        stats = json.load(f)

    print("="*70)
    print("ToS DATASET SUMMARY")
    print("="*70)
    print()

    # Overall stats
    print(f"Total Clauses:    {stats['total_clauses']:,}")
    print(f"Total Companies:  {stats['total_companies']}")
    print()

    # Overall fairness
    print("-" * 70)
    print("OVERALL FAIRNESS DISTRIBUTION")
    print("-" * 70)

    unfair = stats['overall_fairness']['unfair']
    fair = stats['overall_fairness']['fair']
    total = unfair + fair
    unfair_pct = unfair / total * 100
    fair_pct = fair / total * 100

    # ASCII bar chart
    unfair_bar = "█" * int(unfair_pct / 2)
    fair_bar = "█" * int(fair_pct / 2)

    print(f"Unfair:   {unfair:5,} ({unfair_pct:5.2f}%) |{unfair_bar}")
    print(f"Fair:     {fair:5,} ({fair_pct:5.2f}%) |{fair_bar}")
    print()

    # Category statistics
    print("-" * 70)
    print("UNFAIR CATEGORIES DISTRIBUTION")
    print("-" * 70)

    # Sort categories by unfair count
    categories = sorted(
        stats['category_statistics'].items(),
        key=lambda x: x[1]['unfair'],
        reverse=True
    )

    max_unfair = max(cat[1]['unfair'] for cat in categories)
    bar_scale = 50 / max_unfair if max_unfair > 0 else 1

    for category_name, category_stats in categories:
        unfair = category_stats['unfair']
        percentage = category_stats['percentage']
        bar_length = int(unfair * bar_scale)
        bar = "█" * bar_length

        print(f"{category_name:25s} {unfair:4,} ({percentage:5.2f}%) |{bar}")

    print()

    # Multilabel distribution
    print("-" * 70)
    print("MULTILABEL DISTRIBUTION")
    print("-" * 70)

    ml = stats['multilabel_distribution']
    total_clauses = ml['0_categories'] + ml['1_category'] + ml['2_plus_categories']

    zero_pct = ml['0_categories'] / total_clauses * 100
    one_pct = ml['1_category'] / total_clauses * 100
    multi_pct = ml['2_plus_categories'] / total_clauses * 100

    print(f"0 unfair categories:  {ml['0_categories']:5,} ({zero_pct:5.2f}%)")
    print(f"1 unfair category:    {ml['1_category']:5,} ({one_pct:5.2f}%)")
    print(f"2+ unfair categories: {ml['2_plus_categories']:5,} ({multi_pct:5.2f}%)")
    print(f"Max categories:       {ml['max_categories']}")
    print()

    # Load full dataset for additional stats
    df_path = Path(__file__).parent / "tos_dataset.csv"
    df = pd.read_csv(df_path)

    # Company distribution
    print("-" * 70)
    print("COMPANY DISTRIBUTION (Top 15)")
    print("-" * 70)

    company_counts = df['company'].value_counts().head(15)
    max_count = company_counts.max()
    bar_scale = 40 / max_count

    for company, count in company_counts.items():
        bar_length = int(count * bar_scale)
        bar = "█" * bar_length
        print(f"{company:20s} {count:4,} |{bar}")

    print()

    # Example clauses
    print("-" * 70)
    print("SAMPLE UNFAIR CLAUSES (by category)")
    print("-" * 70)

    label_columns = ['Arbitration', 'Choice of Law', 'Content Removal',
                     'Jurisdiction', 'Law', 'Limitation of Liability',
                     'Termination', 'Unilateral Change']

    for category in label_columns:
        unfair_samples = df[df[category] == 1]
        if len(unfair_samples) > 0:
            sample = unfair_samples.iloc[0]
            text = sample['text']
            if len(text) > 150:
                text = text[:147] + "..."
            print(f"\n[{category}]")
            print(f"  Company: {sample['company']}")
            print(f"  Text: {text}")

    print()
    print("="*70)

if __name__ == "__main__":
    print_summary()

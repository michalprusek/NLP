"""Test script to validate ToS classification data loading."""
import pandas as pd
import sys

DATA_PATH = '/Users/michalprusek/Downloads/data'

print("Testing data loading...")
print(f"Data path: {DATA_PATH}\n")

try:
    # Load the dataset splits
    print("1. Loading claudette splits...")
    train_doc_ids = pd.read_csv(f'{DATA_PATH}/claudette_train.tsv', sep='\t')['document'].unique()
    val_doc_ids = pd.read_csv(f'{DATA_PATH}/claudette_val.tsv', sep='\t')['document'].unique()
    test_doc_ids = pd.read_csv(f'{DATA_PATH}/claudette_test.tsv', sep='\t')['document'].unique()
    print(f"   ✓ Train docs: {len(train_doc_ids)}")
    print(f"   ✓ Val docs: {len(val_doc_ids)}")
    print(f"   ✓ Test docs: {len(test_doc_ids)}")

    # Load main dataset
    print("\n2. Loading main ToS dataset...")
    df = pd.read_csv(f'{DATA_PATH}/tos_dataset.csv')
    print(f"   ✓ Total rows: {len(df)}")
    print(f"   ✓ Columns: {list(df.columns)}")

    # Create splits
    print("\n3. Creating dataset splits...")
    df_train = df.loc[df['document'].isin(train_doc_ids)]
    df_val = df.loc[df['document'].isin(val_doc_ids)]
    df_test = df.loc[df['document'].isin(test_doc_ids)]
    print(f"   ✓ Train: {len(df_train)} rows")
    print(f"   ✓ Val: {len(df_val)} rows")
    print(f"   ✓ Test: {len(df_test)} rows")

    # Check label distribution
    print("\n4. Label distribution in training set:")
    df_train_neg = df_train.loc[df_train['label'] == 0]
    df_train_pos = df_train.loc[df_train['label'] == 1]
    print(f"   ✓ Negative (fair): {len(df_train_neg)} ({len(df_train_neg)/len(df_train)*100:.1f}%)")
    print(f"   ✓ Positive (unfair): {len(df_train_pos)} ({len(df_train_pos)/len(df_train)*100:.1f}%)")

    # Check unfairness categories
    print("\n5. Unfairness categories distribution:")
    unfairness_categories = ['A', 'CH', 'CR', 'J', 'LAW', 'LTD', 'TER', 'USE']
    for cat in unfairness_categories:
        cat_count = len(df_train.loc[df_train[cat] == 1])
        print(f"   ✓ {cat}: {cat_count} samples")

    # Sample data point
    print("\n6. Sample clause from dataset:")
    sample = df_train.iloc[0]
    print(f"   Document: {sample['document']}")
    print(f"   Label: {'Unfair' if sample['label'] == 1 else 'Fair'}")
    print(f"   Text preview: {sample['text'][:150]}...")

    print("\n" + "="*50)
    print("✓ All data validation checks passed!")
    print("="*50)

except Exception as e:
    print(f"\n✗ Error: {e}")
    sys.exit(1)

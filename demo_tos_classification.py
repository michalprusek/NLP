"""
Demo script for ToS Classification - shows data setup without API calls.
This demonstrates the notebook functionality without incurring API costs.
"""
import pandas as pd
import re

# Data path
DATA_PATH = '/Users/michalprusek/Downloads/data'

print("=" * 70)
print("ToS Classification Demo")
print("=" * 70)

# Load data
train_doc_ids = pd.read_csv(f'{DATA_PATH}/claudette_train.tsv', sep='\t')['document'].unique()
val_doc_ids = pd.read_csv(f'{DATA_PATH}/claudette_val.tsv', sep='\t')['document'].unique()
test_doc_ids = pd.read_csv(f'{DATA_PATH}/claudette_test.tsv', sep='\t')['document'].unique()

df = pd.read_csv(f'{DATA_PATH}/tos_dataset.csv')
df_train = df.loc[df['document'].isin(train_doc_ids)]
df_train_neg = df_train.loc[df_train['label'] == 0]
df_train_pos = df_train.loc[df_train['label'] == 1]
df_val = df.loc[df['document'].isin(val_doc_ids)]
df_test = df.loc[df['document'].isin(test_doc_ids)]

unfairness_categories = ['A', 'CH', 'CR', 'J', 'LAW', 'LTD', 'TER', 'USE']

# Define legal standards
legal_standards = {
    'A': {'fairness_q': 'Does this clause describe an arbitration dispute resolution process that is not fully optional to the consumer?'},
    'CH': {'fairness_q': 'Does this clause specify conditions under which the service provider could amend and modify the terms of service and/or the service itself?'},
    'CR': {'fairness_q': "Does this clause indicate conditions for content removal in the service provider's full discretion, and/or at any time for any or no reasons and/or without notice nor possibility to retrieve the content."},
    'J': {'fairness_q': "Does this clause state that any judicial proceeding is to be conducted in a place other than the consumer's residence (i.e. in a different city, different country)?"},
    'LAW': {'fairness_q': "Does the clause define the applicable law as different from the law of the consumer's country of residence?"},
    'LTD': {'fairness_q': 'Does this clause stipulate that duties to pay damages by the provider are limited or excluded?'},
    'TER': {'fairness_q': 'Does this clause stipulate that the service provider may suspend or terminate the service at any time for any or no reasons and/or without notice?'},
    'USE': {'fairness_q': 'Does this clause stipulate that the consumer is bound by the terms of use of a specific service, simply by using the service, without even being required to mark that he or she has read and accepted them?'},
}

print("\n1. Dataset Overview")
print("-" * 70)
print(f"Train/Val/Test documents: {len(train_doc_ids)} / {len(val_doc_ids)} / {len(test_doc_ids)}")
print(f"Train/Val/Test clauses: {len(df_train)} / {len(df_val)} / {len(df_test)}")
print(f"Train split: {len(df_train_pos)} unfair / {len(df_train_neg)} fair clauses")
print(f"Imbalance ratio: {len(df_train_neg)/len(df_train_pos):.1f}:1")

print("\n2. Unfairness Categories Distribution (Training Set)")
print("-" * 70)
category_names = {
    'A': 'Arbitration',
    'CH': 'Change terms',
    'CR': 'Content removal',
    'J': 'Jurisdiction',
    'LAW': 'Choice of law',
    'LTD': 'Limitation of liability',
    'TER': 'Termination',
    'USE': 'Unilateral change'
}
for cat in unfairness_categories:
    cat_count = len(df_train.loc[df_train[cat] == 1])
    print(f"  {cat:3s} ({category_names[cat]:25s}): {cat_count:3d} samples")

print("\n3. Sample Clauses")
print("-" * 70)

# Show one unfair and one fair clause
unfair_sample = df_train_pos.iloc[0]
fair_sample = df_train_neg.iloc[0]

print("\nUNFAIR Clause Example:")
print(f"  Document: {unfair_sample['document']}")
print(f"  Categories: {[cat for cat in unfairness_categories if unfair_sample[cat] == 1]}")
print(f"  Text: {unfair_sample['text'][:200]}...")

print("\nFAIR Clause Example:")
print(f"  Document: {fair_sample['document']}")
print(f"  Text: {fair_sample['text'][:200]}...")

print("\n4. Example Evaluation Questions")
print("-" * 70)
for i, (cat, details) in enumerate(list(legal_standards.items())[:3], 1):
    print(f"\n{i}. Category {cat} ({category_names[cat]}):")
    print(f"   Question: {details['fairness_q']}")

print("\n5. How the Classification Works")
print("-" * 70)
print("""
For each clause, the system:
  1. Wraps the clause with a legal expert system prompt
  2. Asks a category-specific question about unfairness
  3. Expects answer starting with "yes" or "no" + justification
  4. Extracts the answer using regex pattern matching
  5. Compares to ground truth labels

Example prompt structure:
  "You are a legal expert on consumer protection law.
   Consider the following online terms of service clause:
   [CLAUSE TEXT]
   [FAIRNESS QUESTION]
   Start your answer with 'yes' or 'no' and justify in ≤50 words."
""")

print("\n6. Evaluation Metrics")
print("-" * 70)
print("""
The notebook computes for each category:
  • Precision: TP / (TP + FP) - accuracy of unfair predictions
  • Recall: TP / (TP + FN) - coverage of actual unfair clauses
  • F1 Score: harmonic mean of precision and recall
  • Confusion matrix: TP, TN, FP, FN counts
""")

print("\n" + "=" * 70)
print("Setup complete! To run experiments:")
print("  1. Open Jupyter: jupyter notebook tos_classification.ipynb")
print("  2. Run cells up to (but not including) the API experiment cell")
print("  3. For the experiment cell: Note it will make API calls (costs money)")
print("  4. Consider testing on a small subset first (modify max_pos_n parameter)")
print("=" * 70)

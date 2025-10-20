from datasets import load_from_disk

ds = load_from_disk('datasets/claudette')

print('=' * 80)
print('DATASET STRUCTURE')
print('=' * 80)
print(f'Splits: {list(ds.keys())}')
print(f'Train: {len(ds["train"])} examples')
print(f'Validation: {len(ds["validation"])} examples')
print(f'Test: {len(ds["test"])} examples')
print()

print('Columns:', ds['train'].column_names)
print()

print('=' * 80)
print('LABEL MAPPING')
print('=' * 80)
labels = [
    ('ltd', 'Limitation of liability'),
    ('ter', 'Unilateral termination'),
    ('ch', 'Unilateral change'),
    ('a', 'Arbitration'),
    ('cr', 'Content removal'),
    ('law', 'Choice of law'),
    ('pinc', 'Other'),
    ('use', 'Contract by using'),
    ('j', 'Jurisdiction')
]
for i, (field, name) in enumerate(labels):
    print(f'{i}: {field:5s} -> {name}')
print()

print('=' * 80)
print('EXAMPLE WITH SINGLE LABEL')
print('=' * 80)
label_fields = ['ltd', 'ter', 'ch', 'a', 'cr', 'law', 'pinc', 'use', 'j']
for ex in ds['train']:
    active = [i for i, f in enumerate(label_fields) if ex[f]]
    if len(active) == 1:
        print(f'Sentence: {ex["sentence"][:150]}')
        print(f'Active labels: {active}')
        for i, f in enumerate(label_fields):
            if ex[f]:
                print(f'  -> Label {i} ({f}): {labels[i][1]}')
        break
print()

print('=' * 80)
print('EXAMPLE WITH MULTIPLE LABELS')
print('=' * 80)
for ex in ds['train']:
    active = [i for i, f in enumerate(label_fields) if ex[f]]
    if len(active) > 1:
        print(f'Sentence: {ex["sentence"][:150]}')
        print(f'Active labels: {active}')
        for i, f in enumerate(label_fields):
            if ex[f]:
                print(f'  -> Label {i} ({f}): {labels[i][1]}')
        break
print()

print('=' * 80)
print('LABEL DISTRIBUTION')
print('=' * 80)
counts = [0] * 9
multi = 0
none = 0
for ex in ds['train']:
    active = [i for i, f in enumerate(label_fields) if ex[f]]
    if len(active) == 0:
        none += 1
    elif len(active) > 1:
        multi += 1
    for i in active:
        counts[i] += 1

total = len(ds['train'])
print(f'Total: {total}')
print(f'No labels: {none} ({100*none/total:.1f}%)')
print(f'Multiple labels: {multi} ({100*multi/total:.1f}%)')
print()
for i in range(9):
    print(f'{i}: {labels[i][1]:30s} {counts[i]:5d} ({100*counts[i]/total:5.1f}%)')


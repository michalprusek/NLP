# ToS Dataset - Converted Format

Dataset převedený z ToS_wInstructions pro multilabel a binary classification úloh.

## Základní informace

- **Celkem clauses**: 9,414
- **Počet společností**: 50
- **Formát**: CSV a JSON
- **Úloha**: Multilabel classification (8 kategorií) + Binary classification (fair/unfair)

## Struktura dat

### Sloupce

- `company`: Název společnosti (např. "Facebook", "Google")
- `clause_id`: Unikátní identifikátor clause (např. "Facebook_1")
- `text`: Text clause z Terms of Service
- **8 kategorií unfair clauses** (0 = fair, 1 = unfair):
  - `Arbitration`: Arbitrážní doložky
  - `Choice of Law`: Volba práva
  - `Content Removal`: Odstranění obsahu
  - `Jurisdiction`: Jurisdikce
  - `Law`: Právní ustanovení
  - `Limitation of Liability`: Omezení odpovědnosti
  - `Termination`: Ukončení služby
  - `Unilateral Change`: Jednostranná změna podmínek
- `is_unfair`: Celková nespravedlnost (0 = fair, 1 = unfair)

## Statistiky

### Overall Fairness
- **Unfair clauses**: 1,032 (10.96%)
- **Fair clauses**: 8,382 (89.04%)

### Kategorie (unfair clauses)
| Kategorie | Unfair | Percentage |
|-----------|--------|------------|
| Limitation of Liability | 296 | 3.14% |
| Termination | 236 | 2.51% |
| Choice of Law | 188 | 2.00% |
| Content Removal | 118 | 1.25% |
| Unilateral Change | 117 | 1.24% |
| Law | 70 | 0.74% |
| Jurisdiction | 68 | 0.72% |
| Arbitration | 44 | 0.47% |

### Multilabel distribuce
- Clauses **bez unfair kategorií**: 8,382 (89.0%)
- Clauses s **1 unfair kategorií**: 937 (10.0%)
- Clauses se **2+ unfair kategoriemi**: 95 (1.0%)
- **Maximální počet** unfair kategorií na jednu clause: 3

### Top 10 společností podle počtu clauses
1. Microsoft: 548 clauses
2. Endomondo: 498 clauses
3. Skype: 455 clauses
4. Airbnb: 391 clauses
5. Headspace: 372 clauses
6. LindenLab: 344 clauses
7. musically: 316 clauses
8. Spotify: 293 clauses
9. eBay: 260 clauses
10. Zynga: 248 clauses

## Soubory

- `tos_dataset.csv`: Dataset v CSV formátu
- `tos_dataset.json`: Dataset v JSON formátu (records orient)
- `dataset_statistics.json`: Podrobné statistiky v JSON formátu
- `README.md`: Tento soubor

## Použití

### Python/Pandas

```python
import pandas as pd

# Načtení CSV
df = pd.read_csv('tos_dataset.csv')

# Načtení JSON
df = pd.read_json('tos_dataset.json')

# Binary classification (fair vs unfair)
X = df['text']
y = df['is_unfair']

# Multilabel classification (8 kategorií)
X = df['text']
label_columns = ['Arbitration', 'Choice of Law', 'Content Removal',
                 'Jurisdiction', 'Law', 'Limitation of Liability',
                 'Termination', 'Unilateral Change']
y = df[label_columns]
```

## Poznámky

- Dataset je **výrazně nevyvážený**: pouze ~11% clauses je označeno jako unfair
- **Multilabel je vzácný**: většina unfair clauses (91%) má pouze 1 unfair kategorii
- **Nejčastější kategorie**: Limitation of Liability (3.14%) a Termination (2.51%)
- Text obsahuje **Penn Treebank tokenizaci**: `-lrb-`, `-rrb-` místo závorek

## Zdroj

Dataset pochází z projektu ToS_wInstructions, který byl použit pro výzkum automatické analýzy Terms of Service dokumentů v rámci projektu CLAUDETTE.

## Licence

Původní dataset ToS_wInstructions - viz zdrojový repozitář.

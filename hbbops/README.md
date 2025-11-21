# HbBoPs: Hyperband-based Bayesian Optimization for Black-box Prompt Selection

Komplexní implementace paperu **"Hyperband-based Bayesian Optimization for Black-box Prompt Selection"** (Schneider et al., 2025) pro GSM8K dataset.

## 📋 Přehled

HbBoPs kombinuje tři klíčové komponenty pro efektivní výběr promptů:

1. **Structural-aware Deep Kernel GP** - pro sample-efficiency (méně evaluací promptů)
2. **Hyperband multi-fidelity scheduler** - pro query-efficiency (méně LLM calls)
3. **Bayesian Optimization** - pro inteligentní výběr kandidátních promptů

## 🏗️ Architektura

### 1. Structural-aware Deep Kernel GP

```
Prompt = Instruction + Few-shot Exemplar
         ↓              ↓
    BERT [CLS]     BERT [CLS]
    (768 dim)      (768 dim)
         ↓              ↓
    Lin(768,64)    Lin(768,64)
       → ReLU         → ReLU
    Lin(64,32)     Lin(64,32)
       → ReLU         → ReLU
         ↓              ↓
    Concat (32 + 32 = 64 dim)
              ↓
         Lin(64,32)
            → ReLU
         Lin(32,10)
              ↓
    Latent Features (10 dim)
              ↓
    GP with ARD Matérn 5/2 kernel
```

**Klíčové vlastnosti:**
- Separátní embeddingy pro instrukce a exempláře
- Učí se low-dimensional (10-dim) latent representation zarovnanou s performance promptů
- Trénuje se online během optimalizace pomocí AdamW (lr=0.01, max_epochs=3000, patience=10)

### 2. Hyperband Scheduler

**Multi-fidelity over validation instances:**

- **Fidelity** = počet validačních instancí použitých k evaluaci promptu
- **bmin** = 10 (minimální počet instancí)
- **η** = 2.0 (halving parameter)

**Příklad Hyperband schedule (nvalid=1319):**

| Bracket (s) | Stage (i) | #Instances (b) | #Prompts (n) |
|-------------|-----------|----------------|--------------|
| 3           | 0         | 10             | 8            |
| 3           | 1         | 20             | 4            |
| 3           | 2         | 40             | 2            |
| 3           | 3         | 80             | 1            |
| 2           | 0         | 20             | 6            |
| 2           | 1         | 40             | 3            |
| ...         | ...       | ...            | ...          |

**Důležité design decisions:**
- ✅ Použití stejných validation instancí pro všechny prompty v stage (paired comparison)
- ✅ Superset struktura: vyšší stages rozšiřují nižší (ne resample)
- ✅ Caching evaluací pro zrychlení
- ✅ Incumbent = prompt s nejnižší val. error mezi těmi evaluovanými na full validation set

### 3. Bayesian Optimization Proposal

- **Acquisition function:** Expected Improvement (EI)
- **Random interleaving:** ρ = 0.1 (hedge against špatné GP predikce)
- GP trénovaný na nejvyšší fidelity level s ≥4 observations

## 📁 Struktura Souborů

```
hbbops/
├── instructions.txt                 # 5 instrukcí pro GSM8K (APE forward mode)
├── examples.txt                     # 50 exemplářů (25 setů × 2 permutace)
├── data/
│   ├── validation.json              # 1319 examples (randomly sampled z train)
│   └── test.json                    # 1319 examples (original test set)
├── hbbops.py                        # Hlavní implementace HbBoPs
├── run_hbbops.py                    # Main script pro spuštění
└── README.md                        # Dokumentace
```

## 🚀 Instalace

```bash
# Závislosti jsou v pyproject.toml
uv sync
```

**Potřebné dependencies:**
- torch
- gpytorch
- transformers
- scipy
- numpy
- datasets

## 💻 Použití

### Spuštění HbBoPs

**Základní použití:**
```bash
cd hbbops
uv run python run_hbbops.py --model Qwen/Qwen2.5-7B-Instruct --backend vllm
```

**S různými parametry:**
```bash
cd hbbops

# Menší model na CPU
uv run python run_hbbops.py \
    --model Qwen/Qwen2.5-3B-Instruct \
    --backend transformers \
    --device cpu \
    --bmin 5 \
    --eta 2.0

# Claude API
uv run python run_hbbops.py \
    --model claude-3-haiku-20240307 \
    --backend claude \
    --encoder bert-base-uncased

# Debug mode
uv run python run_hbbops.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --debug
```

**Parametry:**
- `--model`: Model name nebo path (default: Qwen/Qwen2.5-7B-Instruct)
- `--backend`: Backend pro LLM - `vllm`, `transformers`, `claude` (default: vllm)
- `--bmin`: Minimální počet validation instancí (default: 10)
- `--eta`: Halving parameter pro Hyperband (default: 2.0)
- `--encoder`: Encoder model pro embeddingy (default: bert-base-uncased)
- `--device`: Device - `auto`, `cuda`, `cpu`, `mps` (default: auto)
- `--debug`: Enable debug mode (zobrazí LLM odpovědi a evaluace)
- `--output-dir`: Output directory pro výsledky (default: results)

### 3. Výstupy

**JSON soubor (`results/hbbops_TIMESTAMP.json`):**
```json
{
  "method": "HbBoPs",
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "config": {
    "bmin": 10,
    "eta": 2.0,
    "num_prompts": 250
  },
  "best_prompt": {
    "instruction_id": 2,
    "exemplar_id": 15,
    "instruction": "...",
    "exemplar": "..."
  },
  "validation_error": 0.1234,
  "test_error": 0.1456
}
```

**TXT soubor (`results/hbbops_TIMESTAMP.txt`):**
```
HbBoPs Results
================================================================================

Model: Qwen/Qwen2.5-7B-Instruct
Backend: vllm

Validation error: 0.1234 (12.34%)
Test error: 0.1456 (14.56%)

Best Prompt:
--------------------------------------------------------------------------------
[celý prompt text]
```

## 🔬 Implementační Detaily

### Search Space

**5 instructions × 50 exemplars = 250 candidate prompts**

**Instructions (5):**
- Generované pomocí APE forward mode s Claude 3 Sonnet
- Z 10 input-output příkladů z GSM8K train setu

**Exemplars (50):**
- 25 setů po 5 input-output examples
- Každý set permutován 2× → 50 exemplářů celkem
- Testuje vliv pořadí examples na performance

### Encoder

- **BERT base-uncased** pro embeddingy (768 dim)
- [CLS] token embedding
- Separátně pro instrukce a exempláře

### GP Training

- **Optimizer:** AdamW (lr=0.01)
- **Max epochs:** 3000
- **Early stopping:** patience=10
- **Loss:** Negative log marginal likelihood
- **Data normalization:**
  - Input: Z-score normalization (zero mean, unit variance)
  - Output: Z-score normalization

### Evaluace Promptů

**Answer Extraction (priority order):**
1. `final_answer: NUMBER` pattern
2. `#### NUMBER` pattern (GSM8K format)
3. `\boxed{NUMBER}` pattern (LaTeX)
4. Last number in text (fallback)

**Scoring:**
- Exact match s numerical tolerance (1e-4)
- Validation error = fraction of incorrect answers

## 📊 Expected Performance

Podle paperu (průměr přes 10 benchmarks a 3 LLMs):
- **HbBoPs:** 0.150 normalized test error @ full budget
- **TRIPLE-SH (nejbližší konkurent):** 0.159
- **Improvement:** ~6% lepší než TRIPLE-SH

**Anytime performance:**
- @ 0.25 budget: 24% lepší než TRIPLE-SH
- @ 0.50 budget: 21% lepší než TRIPLE-SH

## 🎯 Klíčové Výhody HbBoPs

1. **Sample-efficient:** Structural-aware DK-GP efektivně exploruje search space
2. **Query-efficient:** Hyperband rychle filtruje špatné prompty s malým počtem instancí
3. **Both:** Kombinace BO + Hyperband je sample- i query-efficient
4. **Structural awareness:** Separátní embeddingy zachycují odlišnou strukturu instrukcí vs. exemplářů

## 🔧 Troubleshooting

**Out of Memory:**
```bash
# Použij menší model
--model Qwen/Qwen2.5-3B-Instruct

# Použij CPU
--device cpu --backend transformers

# Snížit bmin (méně validation instancí)
--bmin 5
```

**Slow training:**
```bash
# Použij vLLM místo transformers
--backend vllm

# Použij GPU
--device cuda
```

**GP training issues:**
```bash
# Debug mode ukáže GP training progress
--debug
```

## 📚 Reference

```
@inproceedings{schneider2025hbbops,
  title={Hyperband-based Bayesian Optimization for Black-box Prompt Selection},
  author={Schneider, Lennart and Wistuba, Martin and Klein, Aaron and
          Golebiowski, Jacek and Zappella, Giovanni and Merra, Felice Antonio},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning},
  year={2025}
}
```

## 📝 Notes

- **Reproducibility:** Random seed je fixován v setup scriptu (seed=42)
- **Validation split:** 1319 examples náhodně samplovány z train setu (stejná velikost jako test)
- **Caching:** LLM outputs jsou cachovány - restart nepřehodnotí již evaluované prompty
- **Multi-GPU:** Pro vLLM backend lze použít `--tensor-parallel-size` v llm_client.py

## 🐛 Known Limitations

1. **Small encoder models:** BERT base má omezení na 512 tokenů - dlouhé exempláře jsou truncated
2. **GP scaling:** Pro >1000 observations může být GP training pomalý
3. **Memory:** Ukládání všech embeddings v paměti může být problém pro velké search spaces

## 🚧 Future Work

- [ ] Podpora pro více encoder models (MPNet, DistilRoBERTa)
- [ ] Parallelizace Hyperband brackets
- [ ] Support pro další datasets (ARC, BBII tasks)
- [ ] Vizualizace GP latent space (t-SNE)
- [ ] Comparison s baseline methods (EASE, MIPROv2, TRIPLE)

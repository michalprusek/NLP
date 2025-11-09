# Prompt Optimization Visualization

Interaktivní vizualizace trajektorií optimalizace promptů pomocí embeddingů a UMAP redukce dimenzionality.

## Funkce

- **Lokální embeddingy**: Použití sentence-transformers pro převod promptů na vektory
- **UMAP redukce**: Redukce vysokodimenzionálních embeddingů do 2D/3D prostoru
- **Interaktivní grafy**: Plotly vizualizace s hover informacemi
- **Dvojí obarvení**: Každý graf v 2 verzích
  - **Accuracy heatmapa**: Barvy bodů podle dosažené accuracy (červená → zelená)
  - **Iterační timeline**: Barvy bodů podle iterace (fialová → zelená → žlutá)
- **Iterační značky**: Viditelné značení iterací optimalizace
- **4 grafy celkem**: 2D accuracy, 2D iteration, 3D accuracy, 3D iteration

## Instalace závislostí

```bash
uv sync
```

Nové dependencies:
- `sentence-transformers`: Lokální model pro text embeddings
- `umap-learn`: Algoritmus pro redukci dimenzionality
- `plotly`: Interaktivní vizualizace
- `pandas`: Manipulace s daty

## Použití

### Základní použití

```bash
# 2D a 3D vizualizace (výchozí)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json

# Pouze 2D
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json --dimensions 2d

# Pouze 3D
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json --dimensions 3d
```

### Pokročilé parametry

**Embedding model**:
```bash
# Rychlý model (výchozí) - 384 dimenzí
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --embedding-model all-MiniLM-L6-v2

# Lepší kvalita - 768 dimenzí
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --embedding-model all-mpnet-base-v2

# Multilingvální model
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --embedding-model paraphrase-multilingual-MiniLM-L12-v2
```

**UMAP parametry**:
```bash
# Více lokální struktury (menší n_neighbors)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --n-neighbors 5 --min-dist 0.05

# Více globální struktury (větší n_neighbors)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --n-neighbors 50 --min-dist 0.2

# Tužší shluky (menší min_dist)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --min-dist 0.01

# Rozptýlenější body (větší min_dist)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --min-dist 0.5
```

**Metriky vzdálenosti**:
```bash
# Cosine similarity (výchozí, doporučeno pro text)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --metric cosine

# Euklidovská vzdálenost
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --metric euclidean

# Manhattan vzdálenost
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --metric manhattan
```

**Hardware**:
```bash
# CPU (výchozí)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json --device cpu

# NVIDIA GPU
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json --device cuda

# Apple Silicon GPU
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json --device mps
```

**Vlastní výstupní složka**:
```bash
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --output-dir my_visualizations
```

## Výstup

Skript vytvoří v `visualizations/output/` (nebo vlastní složce) HTML soubory:
- `{název_souboru}_2d_accuracy.html`: 2D scatter plot obarvený podle accuracy
- `{název_souboru}_2d_iteration.html`: 2D scatter plot obarvený podle iterace
- `{název_souboru}_3d_accuracy.html`: 3D scatter plot obarvený podle accuracy
- `{název_souboru}_3d_iteration.html`: 3D scatter plot obarvený podle iterace

### Interakce s grafy

- **Hover**: Zobrazí prompt text, accuracy a iteraci
- **Zoom**: Kolečko myši nebo pinch
- **Pan**: Kliknutí a tažení
- **3D rotace**: Tažení
- **Reset**: Dvojklik na graf

## Interpretace vizualizace

### Barvy - Accuracy grafy (_accuracy.html)
- 🔴 **Červená**: Nízká accuracy (~0.3-0.6)
- 🟡 **Žlutá**: Střední accuracy (~0.6-0.8)
- 🟢 **Zelená**: Vysoká accuracy (~0.8-0.95)

### Barvy - Iteration grafy (_iteration.html)
- 🟣 **Fialová**: Časné iterace (0-3)
- 🟢 **Zelená**: Střední iterace (4-6)
- 🟡 **Žlutá**: Pozdní iterace (7-9)

### Struktura
- **Clustery**: Podobné prompty (sémanticky) jsou blízko u sebe
- **Trajektorie**: Sledujte značky "I0", "I1", ... pro průběh iterací
- **Outliers**: Prompty daleko od ostatních jsou sémanticky unikátní

### Co hledat v Accuracy grafech
1. **Konvergence**: Shluky zelených bodů = optimalizace našla dobrou oblast
2. **Explorace**: Rozptýlené body = algoritmus zkoušel různé strategie
3. **Skok v kvalitě**: Výrazná změna barvy mezi iteracemi
4. **Dead ends**: Červené body daleko od hlavního clusteru

### Co hledat v Iteration grafech
1. **Časová evoluce**: Jak se prompty vyvíjely od fialové (start) k žluté (konec)
2. **Explorace vs exploitace**: Fialové body rozptýlené = počáteční explorace
3. **Konvergence**: Žluté body v jednom clusteru = konvergence k řešení
4. **Návrat do oblastí**: Kdy se algoritmus vrací do dříve navštívených oblastí

## Doporučené embedding modely

| Model | Dimenze | Rychlost | Kvalita | Použití |
|-------|---------|----------|---------|---------|
| all-MiniLM-L6-v2 | 384 | ⚡⚡⚡ | ⭐⭐ | Rychlá explorace |
| all-mpnet-base-v2 | 768 | ⚡⚡ | ⭐⭐⭐ | Lepší sémantická reprezentace |
| all-MiniLM-L12-v2 | 384 | ⚡⚡ | ⭐⭐⭐ | Dobrý kompromis |

## UMAP parametry - průvodce

### n_neighbors (výchozí: 15)
- **5-10**: Zdůrazní lokální strukturu, více clusterů
- **15-30**: Balanced (doporučeno)
- **30-100**: Zachová globální strukturu, méně clusterů

### min_dist (výchozí: 0.1)
- **0.0**: Velmi těsné clustery, body na sobě
- **0.1**: Balanced (doporučeno)
- **0.5+**: Uniformní rozložení, méně clusterů

### metric
- **cosine**: Nejlepší pro text embeddings (doporučeno)
- **euclidean**: Geometrická vzdálenost
- **manhattan**: L1 vzdálenost

## Příklady workflow

### Rychlá explorace
```bash
# Základní vizualizace (vytvoří 2 grafy: accuracy + iteration)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json --dimensions 2d
```

### Plná analýza (doporučeno)
```bash
# Lepší model + všechny grafy (vytvoří 4 grafy)
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --embedding-model all-mpnet-base-v2 \
    --dimensions both \
    --device cpu
```

### Citlivější UMAP parametry
```bash
# Pro detailnější strukturu
uv run python visualizations/visualize_prompts.py results/opro_20251019_011854.json \
    --embedding-model all-mpnet-base-v2 \
    --n-neighbors 10 \
    --min-dist 0.05 \
    --device mps
```

### Porovnání různých běhů
```bash
# Vytvoř vizualizace pro více experimentů
for file in results/opro_*.json; do
    uv run python visualizations/visualize_prompts.py "$file" \
        --embedding-model all-mpnet-base-v2 \
        --dimensions both
done
```

### Jak grafy používat společně
1. **Začni s _iteration.html**: Zjisti, jak algoritmus exploroval prompt space
2. **Přepni na _accuracy.html**: Identifikuj oblasti s vysokou kvalitou
3. **Kombinuj poznatky**: Kdy našel dobré oblasti? Vrátil se do nich?
4. **3D pro detaily**: Když 2D není dostatečně čitelné

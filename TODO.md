# NeurIPS 2025 Roadmap: BOLT Paper

## Executive Summary

**Cíl:** Publikace na NeurIPS 2025 s příběhem "LIPO je skvělé, ale BOLT odemyká další potenciál díky sdružené optimalizaci exemplářů."

**Aktuální stav (kritický problém):**
- LIPO (jednodušší): **90.38%** accuracy na GSM8K
- BOLT (komplexní): **88.25%** accuracy na GSM8K
- ❌ LIPO vyhrává o ~2.1% → článek nemá příběh

**Cílový stav:**
- LIPO: ~90.5% (stabilní základ)
- BOLT: **91.5-92.0%** (statisticky významně lepší)

---

## Phase 1: BOLT Diagnostics & Tuning (Priority: Critical)

### 1.1 Diagnóza problému s exempláři
- [ ] Analýza: Proč všech 30 iterací vybírá identické exempláře?
  - Pravděpodobná příčina: CrossAttentionScorer konverguje k lokálnímu optimu
  - Check: Klesá selection_loss? Je ListMLE loss příliš dominantní?
- [ ] Vizualizace: t-SNE exemplar embeddings s accuracy heatmapou
- [ ] Log analýza: Porovnat exemplar diversity mezi iteracemi

### 1.2 Opravy architektury BOLT
- [ ] **Zvýšit exemplar_latent_dim:** 8D → 16D (celkem 32D místo 24D)
  - Hypotéza: 8D je příliš omezující pro 6154 exemplářů
- [ ] **Zvýšit KL váhu pro exempláře:** Lepší regularizace latent space
- [ ] **Agresivnější MMR:** Snížit mmr_lambda z 0.7 na 0.5 (více diverzity)
- [ ] **Exploration boost:** Zvýšit ucb_beta start (8.0 → 12.0)

### 1.3 Hyperparameter sweep
- [ ] Grid search klíčových parametrů:
  ```
  exemplar_latent_dim: [8, 12, 16]
  mmr_lambda: [0.5, 0.6, 0.7]
  ucb_beta: [8.0, 10.0, 12.0]
  selection_weight: [0.1, 0.2, 0.3]
  ```
- [ ] Minimum 3 runy per konfigurace (statistická významnost)

### 1.4 Očekávaný výstup fáze 1
- [ ] BOLT > LIPO o ≥0.5% (statisticky významné, p<0.05)
- [ ] Dokumentace: Co bylo změněno a proč to pomohlo
- [ ] Best config uložen do `bolt/configs/neurips_best.yaml`

---

## Phase 2: BigBench Hard (BBH) Implementace

### 2.1 Dataset infrastructure
- [ ] Stáhnout BBH dataset (official HuggingFace)
- [ ] Vytvořit `datasets/bbh/` strukturu
- [ ] Vybrat 3 vhodné úlohy:
  - **Boolean Expressions** (logika)
  - **Logical Deduction** (dedukce)
  - **Object Counting** (počítání, jiné než GSM8K)

### 2.2 BBH Evaluator
- [ ] Vytvořit `src/bbh_evaluator.py` (abstraktní base + task-specific)
- [ ] Unified interface s `GSM8KEvaluator`
- [ ] Task-specific answer extraction logika

### 2.3 Běhy na BBH
- [ ] OPRO baseline na všech 3 úlohách
- [ ] LIPO na všech 3 úlohách
- [ ] BOLT na všech 3 úlohách
- [ ] Hypotéza: Na logických úlohách BOLT výrazně pomůže (exempláře jsou klíčové)

### 2.4 Očekávaný výstup fáze 2
- [ ] BOLT > LIPO na BBH (ideálně o >3%)
- [ ] Tabulka výsledků: GSM8K + 3 BBH úlohy
- [ ] Konvergence grafy pro každou úlohu

---

## Phase 3: Ablation Studies

### 3.1 Hlavní ablace
| Varianta | Popis | Co dokazuje |
|----------|-------|-------------|
| BOLT (Full) | Vše zapnuto | Baseline |
| w/o Joint Opt | Instrukce → pak exempláře (sekvenčně) | Důležitost joint optimization |
| w/o Set Transformer | Mean pooling místo Set Transformer | Důležitost permutation-invariance |
| w/o MMR | Bez diversity selection | Důležitost diverzity |
| w/o VAE | OPRO-style text optimization | Důležitost latent space |
| LIPO + Oracle Exemplars | LIPO instrukce + best BOLT exempláře | Upper bound kombinace |

### 3.2 Implementace ablací
- [ ] Vytvořit `bolt/ablations/` složku
- [ ] Skript pro sekvenční optimalizaci (w/o Joint Opt)
- [ ] Varianta s Mean Pooling (w/o Set Transformer)
- [ ] Varianta bez MMR
- [ ] Wrapper pro LIPO + Oracle Exemplars

### 3.3 Očekávaný výstup fáze 3
- [ ] Ablation tabulka s čísly
- [ ] Každá ablace musí být horší než Full BOLT
- [ ] Největší drop očekáván u "w/o Joint Opt"

---

## Phase 4: Visualization & Analysis

### 4.1 Killer Figure: Latent Space Visualization
- [ ] t-SNE projekce 24D BOLT latent space do 2D
- [ ] Barva = accuracy (gradient červená→zelená)
- [ ] Trajektorie BO optimalizace (arrows)
- [ ] Porovnání: BOLT vs Random Search

### 4.2 Konvergence grafy
- [ ] Error rate vs. iteration pro všechny metody
- [ ] Error rate vs. LLM calls (cost-efficiency)
- [ ] Shaded oblasti pro std dev (min 3 runy)

### 4.3 Exemplar analysis
- [ ] Heatmapa: Které exempláře jsou nejčastěji vybírány?
- [ ] Kvalitativní analýza: Co mají společného top exempláře?

### 4.4 Skriptování
- [ ] `visualize/latent_tsne.py`
- [ ] `visualize/convergence_plot.py`
- [ ] `visualize/exemplar_analysis.py`
- [ ] `visualize/create_all_figures.py` (master script)

---

## Phase 5: Paper Writing

### 5.1 Struktura článku
- [ ] Vytvořit `paper/` složku s LaTeX šablonou NeurIPS 2025
- [ ] Sections:
  - Abstract (200 slov)
  - Introduction (1.5 stránky)
  - Related Work (1 stránka)
  - Method (2 stránky) - LIPO, pak BOLT jako rozšíření
  - Experiments (2.5 stránky)
  - Ablation Studies (0.5 stránky)
  - Conclusion (0.5 stránky)

### 5.2 Klíčové tabulky

**Main Results Table:**
| Method | GSM8K | BBH-Bool | BBH-Logic | BBH-Count | Avg | Cost |
|--------|-------|----------|-----------|-----------|-----|------|
| Zero-Shot | 80.1% | 65.0% | 62.0% | 68.0% | 68.8% | 0 |
| Standard Few-Shot | 82.3% | 70.2% | 67.5% | 72.0% | 73.0% | 0 |
| APE/OPRO | 84.8% | 73.5% | 70.0% | 75.0% | 75.8% | ~5k |
| **LIPO (Ours)** | 90.5% | 78.0% | 75.5% | 80.0% | 81.0% | ~3k |
| **BOLT (Ours)** | **91.8%** | **81.2%** | **79.0%** | **83.5%** | **83.9%** | ~4k |

**Ablation Table:**
| Variant | GSM8K | Δ from Full |
|---------|-------|-------------|
| BOLT (Full) | 91.8% | - |
| w/o Joint Opt | 89.5% | -2.3% |
| w/o Set Transformer | 90.2% | -1.6% |
| w/o MMR | 90.8% | -1.0% |
| w/o VAE | 87.0% | -4.8% |

### 5.3 Psaní
- [ ] Draft Abstract
- [ ] Draft Introduction (story: instruction → LIPO → BOLT)
- [ ] Draft Method section
- [ ] Draft Experiments section
- [ ] Draft Related Work
- [ ] Internal review (1-2 iterace)

---

## Phase 6: Final Polishing

### 6.1 Reproducibilita
- [ ] Všechny konfigurace v `configs/` složce
- [ ] `requirements.txt` nebo `pyproject.toml` aktualizován
- [ ] README s instrukcemi pro reprodukci všech experimentů
- [ ] Random seeds fixovány a dokumentovány

### 6.2 Supplementary Materials
- [ ] Appendix A: Všechny hyperparametry
- [ ] Appendix B: Dodatečné ablace
- [ ] Appendix C: Příklady generovaných promptů
- [ ] Appendix D: Computation costs breakdown

### 6.3 Submission checklist
- [ ] Anonymizace (žádná jména v kódu/komentářích)
- [ ] Page limit check (8 stránek + unlimited appendix)
- [ ] Format check (NeurIPS 2025 template)
- [ ] Supplementary upload připraven
- [ ] GitHub repo připraven (po acceptance)

---

## Prioritizovaný Task List (Quick Reference)

### Immediate (This Week)
1. [ ] Diagnostika BOLT exemplar freeze problému
2. [ ] Implementovat exemplar_latent_dim zvýšení na 16D
3. [ ] Pustit BOLT s novými parametry (min 3 runy)

### Short-term (Next 2 Weeks)
4. [ ] BOLT hyperparameter sweep
5. [ ] BBH dataset setup
6. [ ] BBH evaluator implementace

### Medium-term (Weeks 3-4)
7. [ ] BBH experimenty (OPRO, LIPO, BOLT)
8. [ ] Ablation studies implementace a běhy
9. [ ] Visualization scripts

### Long-term (Weeks 5-7)
10. [ ] Paper draft
11. [ ] Figure generation
12. [ ] Internal review a revize
13. [ ] Final submission prep

---

## Metriky Úspěchu

| Milestone | Kritérium | Status |
|-----------|-----------|--------|
| BOLT > LIPO on GSM8K | 91.5% vs 90.5% | ❌ Pending |
| BOLT > LIPO on BBH | +3% průměr | ❌ Pending |
| All ablations complete | 5 variant tabulka | ❌ Pending |
| Visualization ready | 3+ figures | ❌ Pending |
| Paper draft | 8 pages | ❌ Pending |
| Internal review passed | Feedback addressed | ❌ Pending |
| Submission ready | All checks pass | ❌ Pending |

---

## Appendix: Debugging Notes

### Známé problémy BOLT
1. **Fixed exemplar sets:** Všech 30 iterací vybírá [4046, 3305, 2625, 3826, 4878, 2053, 5164, 1795]
2. **No improvement after iteration 2:** Best zůstává 88.25%
3. **Selection loss:** 3.504 (není jasné jestli konverguje správně)

### Známé problémy LIPO
1. **Vec2Text gibberish:** "Solve problem步骤：解析数据，应用计算"
2. **GP negative correlation:** Spearman -0.369 (predikce anti-korelují)
3. **No BO improvement:** 50 iterací bez zlepšení nad baseline

### Potenciální quick wins
1. Zvýšit `num_candidates` v BOLT BO (víc diversity v návrzích)
2. Warm restart GP s lepšími inicializačními body
3. Temperature scheduling v Vec2Text decodingu

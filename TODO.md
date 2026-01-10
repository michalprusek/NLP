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

## Phase 1: BOLT Component-Wise Optimization (Priority: Critical)

> **Strategie:** Coordinate Descent místo slepého Grid Search. Optimalizujeme subsystémy izolovaně, abychom identifikovali bottleneck.

### 1.1 VAE Latent Space Tuning (Rekonstrukce)

**Cíl:** Ověřit, že VAE umí reprezentovat exempláře věrně.

- [ ] **Retrieval Accuracy @ K test:**
  ```python
  # Test: 8 náhodných exemplářů → encode → decode → retrieve z poolu
  # Metrika: Kolik z původních 8 se vrátí v top-8?
  for _ in range(100):  # 100 random trials
      orig_8 = random.sample(exemplar_pool, 8)
      z_ex = vae.encode(orig_8)
      decoded = vae.decode(z_ex)
      retrieved = nearest_neighbors(decoded, exemplar_pool, k=8)
      retrieval_acc = len(set(orig_8) & set(retrieved)) / 8
  ```
- [ ] **Cílová metrika:** Retrieval Accuracy @ 8 ≥ 0.85
- [ ] Hyperparametry k ladění:
  - `exemplar_latent_dim`: 8 → **16 nebo 32** (8D je podezřele málo)
  - `set_transformer_heads`: 4 → **8** (více attention kapacity)
  - `set_transformer_depth`: zvážit přidání vrstev

### 1.2 CrossAttention Scorer Tuning (Ranking)

**Cíl:** Scorer musí umět správně seřadit exempláře PŘEDTÍM, než ho zapojíme do BO.

- [ ] **Synthetic Retrieval Task (supervised pre-training):**
  ```python
  # 1. Pro každou instrukci najdi "Golden Exemplars" hrubou silou
  for instruction in train_instructions:
      inst_emb = gtr.encode(instruction)
      golden_8 = top_k_cosine_sim(inst_emb, exemplar_pool, k=8)

  # 2. Trénuj Scorer supervised, aby predikoval Golden Exemplars
  scorer.train(instruction_embeddings, golden_exemplar_sets)

  # 3. Měř NDCG@8 nebo MRR na held-out setu
  ```
- [ ] **Cílová metrika:** NDCG@8 ≥ 0.7 na validation set
- [ ] **Důležité:** Scorer musí být přesný PŘED BO smyčkou, ne se učit "za běhu"

### 1.3 GP Kernel Tuning (Prior na Exempláře)

**Cíl:** Donutit GP věřit, že exempláře MAJÍ vliv na accuracy.

- [ ] **ARD Lengthscale Priors:**
  ```python
  # Problém: GP ignoruje exemplar dimenze → zamrzlé exempláře
  # Řešení: Vynuť menší lengthscale pro exemplar dimenze

  # Dimenze 0-15 (instrukce): standardní prior
  instruction_prior = gpytorch.priors.GammaPrior(2.0, 2.0)  # mean ~1.0

  # Dimenze 16-31 (exempláře): agresivnější prior
  exemplar_prior = gpytorch.priors.GammaPrior(4.0, 4.0)  # mean ~1.0, ale užší
  # NEBO přímo constraint na menší hodnoty:
  exemplar_constraint = gpytorch.constraints.Interval(0.1, 0.5)
  ```
- [ ] **Efekt:** Malá změna v z_ex → velká změna v predikované accuracy → BO exploruje exempláře
- [ ] **Test:** Po 10 BO iteracích musí být variance v exemplar selections > 0

### 1.4 Diagnóza (původní úkoly)
- [ ] Analýza: Proč všech 30 iterací vybírá identické exempláře?
- [ ] Log: Sledovat `selection_loss` trend - klesá správně?
- [ ] Vizualizace: t-SNE exemplar embeddings s accuracy heatmapou

### 1.5 Orchestrace ladění
- [ ] **Pořadí:** VAE → Scorer → GP (každý krok musí projít před dalším)
- [ ] **Checkpoint:** Po každém kroku uložit metriky do `bolt/tuning_log.json`
- [ ] Minimum 3 runy per konfigurace (statistická významnost)

### 1.6 Očekávaný výstup fáze 1
- [ ] VAE Retrieval Accuracy @ 8 ≥ 0.85
- [ ] Scorer NDCG@8 ≥ 0.7
- [ ] GP exemplar variance > 0 po 10 iteracích
- [ ] **BOLT > LIPO o ≥0.5%** (statisticky významné, p<0.05)
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

## Phase 3: Ultimate Ablation Study

> **Cíl:** Rozbít systém na prvočinitele. Izolovat přesný přínos každé komponenty. Recenzenti to milují.

### 3.1 Hlavní ablační tabulka (POVINNÁ pro paper)

| # | Metoda | Co optimalizujeme? | Jak vybíráme to druhé? | Story / Hypotéza |
|---|--------|-------------------|------------------------|------------------|
| A | **Zero-Shot** | Nic | - | Lower bound baseline |
| B | **Exemplar Search Only** | Exempláře | Fixní instrukce ("Solve step by step") | "Stačí jen najít dobré příklady?" |
| C | **LIPO (Instruction Only)** | Instrukce | Žádné exempláře (nebo fixní 8-shot) | "Stačí jen lepší instrukce?" |
| D | **Sequential Opt** | Instrukce → Exempláře | Najdi LIPO instrukci, pak k ní dohledej ex. | "Musíme to dělat najednou?" |
| E | **BOLT (Joint)** | Instrukce + Exempláře | Společně v latentním prostoru | "Synergie je klíč." |

**Očekávaná hierarchie:** A < B < C < D < E

**Interpretace výsledků:**
- Pokud **E ≤ D**: Joint optimization selhává (kritický bug)
- Pokud **C > B**: Instrukce je důležitější než exempláře (typické pro LLM)
- Pokud **E >> D**: Joint optimization přináší synergii (BOLT story)

### 3.2 Komponentní ablace (doplňkové)

| Varianta | Popis | Co dokazuje |
|----------|-------|-------------|
| BOLT w/o Set Transformer | Mean pooling místo Set Transformer | Důležitost permutation-invariance |
| BOLT w/o MMR | Bez diversity selection | Důležitost diverzity exemplářů |
| BOLT w/o VAE | OPRO-style text optimization | Důležitost latent space |
| BOLT w/o DKL | Standardní GP místo Deep Kernel | Důležitost learned features |
| LIPO + Oracle Ex. | LIPO instrukce + best BOLT exempláře | Upper bound kombinace |

### 3.3 Implementace

- [ ] Vytvořit `bolt/ablations/` složku
- [ ] **A: Zero-Shot** - triviální baseline
- [ ] **B: Exemplar Search Only:**
  ```python
  # Fixní instrukce: "Let's solve this step by step."
  # Optimalizuj jen výběr 8 exemplářů pomocí GP
  ```
- [ ] **C: LIPO** - již implementováno
- [ ] **D: Sequential Opt:**
  ```python
  # 1. Spusť LIPO, získej best_instruction
  # 2. Pro best_instruction najdi optimální exempláře (GP search)
  # 3. Kombinuj a vyhodnoť
  ```
- [ ] **E: BOLT (Joint)** - již implementováno
- [ ] Komponenty: Mean Pooling variant, No-MMR variant, No-DKL variant

### 3.4 Statistická validace

- [ ] Minimum **5 runů** per ablace (pro robustní std dev)
- [ ] **Paired t-test** nebo **Wilcoxon** pro E vs D srovnání
- [ ] Report: mean ± std, p-values
- [ ] Effect size (Cohen's d) pro hlavní srovnání

### 3.5 Očekávaný výstup fáze 3

- [ ] Kompletní 5-řádková ablation tabulka (A-E)
- [ ] Hierarchie A < B < C < D < E potvrzena
- [ ] **E - D ≥ 1.0%** (statisticky významné, p<0.05)
- [ ] Komponenty: každá horší než Full BOLT

---

## Phase 4: Visualization & Analysis

### 4.1 Killer Figure: Latent Space Trajectory (UMAP)
- [ ] **UMAP projekce** 24D BOLT latent space do 2D (preferuj UMAP před t-SNE pro trajektorie)
- [ ] Barva = accuracy (gradient červená→zelená)
- [ ] **Trajektorie BO optimalizace** (šipky spojující iterace)
- [ ] Porovnání: BOLT trajectory vs Random Search (scatter bez struktury)
- [ ] Anotace: Start point, best found, exploration vs exploitation fáze

### 4.2 Sensitivity Analysis Heatmap (KLÍČOVÁ pro BOLT story)

**Cíl:** Ukázat, že záleží na OBOU osách (instrukce i exempláře), ne jen na jedné.

- [ ] **UMAP 2D projekce:**
  ```python
  # Osa X: UMAP dim 1 z instruction latent (z_inst)
  # Osa Y: UMAP dim 1 z exemplar latent (z_ex)
  # Barva: GP-predikovaná accuracy (nebo skutečná, pokud máme)
  ```
- [ ] **Interpretace vzorů:**
  | Vzor | Význam | Implikace |
  |------|--------|-----------|
  | Svislé pruhy | Záleží jen na instrukci | BOLT nepomáhá, LIPO stačí |
  | Vodorovné pruhy | Záleží jen na exemplářích | Instrukce je irelevantní |
  | Diagonální hřebeny | Interakce inst×ex | BOLT dává smysl |
  | **Ostrovy** | Specifické páry fungují | **BOLT story - synergie** |
- [ ] **Aktuální hypotéza:** Vidíme svislé pruhy → proto LIPO vyhrává

### 4.3 Konvergence grafy
- [ ] Error rate vs. iteration pro všechny metody (A-E z ablací)
- [ ] Error rate vs. LLM calls (cost-efficiency, sample complexity)
- [ ] Shaded oblasti pro std dev (min 3 runy)
- [ ] Log-scale x-axis pro zobrazení early gains

### 4.4 Exemplar Analysis
- [ ] **Frequency heatmap:** Které exempláře jsou nejčastěji vybírány?
- [ ] **Exemplar clustering:** UMAP exemplar embeddings, obarvené podle frequency
- [ ] Kvalitativní analýza: Co mají společného top exempláře?
  - Délka? Struktura? Typ problému?
- [ ] **Diversity over iterations:** Měří se, jak moc se exemplar sets mění

### 4.5 GP Lengthscale Analysis
- [ ] **Bar chart:** Learned ARD lengthscales per dimension
- [ ] Očekávání: Instruction dims mají menší lengthscale (více důležité)
- [ ] **Po opravě:** Exemplar dims by měly mít comparable lengthscales

### 4.6 Smoothness Analysis (Teoretická obhajoba)

**Motivace:** NeurIPS recenzent se zeptá: "Je latentní prostor hladký? Když se pohnu o ε v z, změní se výstup o ε, nebo o kilometr?"

> **Důležité:** Pokud je error landscape chaotický ("rozbouřené moře"), BO nemůže fungovat. Potřebujeme "hladké údolí".

- [ ] **Lipschitz konstanta (přibližná):**
  ```python
  # Měření pro decoder i scorer
  def estimate_lipschitz(model, z_samples, epsilon=0.01, n_directions=100):
      """Aproximace Lipschitz konstanty pomocí náhodných perturbací."""
      lipschitz_estimates = []
      for z in z_samples:
          for _ in range(n_directions):
              direction = torch.randn_like(z)
              direction = direction / direction.norm() * epsilon

              z_perturbed = z + direction

              # Pro decoder: měř embedding distance
              out_orig = model.decode(z)
              out_pert = model.decode(z_perturbed)
              output_change = cosine_distance(out_orig, out_pert)

              # Lipschitz = output_change / input_change
              lipschitz_estimates.append(output_change / epsilon)

      return max(lipschitz_estimates), np.mean(lipschitz_estimates)
  ```
- [ ] **Cílové hodnoty:**
  - Lipschitz < 10: Hladký prostor (BO bude fungovat)
  - Lipschitz > 100: Příliš strmý (BO selhává)

- [ ] **Error Landscape Smoothness graf:**
  ```python
  # Vizualizace: 2D řez error landscape
  # 1. Vezmi best_z a 2 náhodné ortogonální směry
  # 2. Udělej grid [-2σ, +2σ] v obou směrech
  # 3. Pro každý bod: decode → evaluate → error_rate
  # 4. Contour plot
  ```
  - **Očekávaný výsledek:** Hladké kontury, ne "šachovnice"

- [ ] **Fix pokud není hladký:**
  - Zvýšit `vae_beta` (KL divergence) → vynucuje hladší latent space
  - Test: `vae_beta` ∈ [0.01, 0.02, 0.05, 0.1]
  - Trade-off: Vyšší beta = hladší, ale horší rekonstrukce

- [ ] **Report pro paper:**
  - Tabulka: Lipschitz konstanta pro instruction vs exemplar subspace
  - Graf: Error landscape contour plot

### 4.7 Skriptování
- [ ] `visualize/latent_umap.py` - UMAP trajectory + heatmap
- [ ] `visualize/sensitivity_heatmap.py` - inst×ex interaction plot
- [ ] `visualize/convergence_plot.py` - error rate curves
- [ ] `visualize/exemplar_analysis.py` - frequency + clustering
- [ ] `visualize/lengthscale_analysis.py` - GP diagnostics
- [ ] `visualize/smoothness_analysis.py` - Lipschitz + error landscape
- [ ] `visualize/create_all_figures.py` - master script pro paper

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

### Immediate: Coordinate Descent Phase (This Week)
> **Pravidlo:** Každý krok MUSÍ projít před dalším. Neskok na další, dokud není metrika splněna.

1. [ ] **VAE Retrieval Test:** Implementovat Retrieval Accuracy @ 8 metriku
2. [ ] **VAE Tuning:** exemplar_latent_dim 8→16, set_transformer_heads 4→8
3. [ ] **VAE Checkpoint:** Retrieval Acc ≥ 0.85 ✓

4. [ ] **Scorer Pre-training:** Synthetic Retrieval Task (Golden Exemplars)
5. [ ] **Scorer Checkpoint:** NDCG@8 ≥ 0.7 ✓

6. [ ] **GP Prior Fix:** ARD lengthscale priors pro exemplar dimenze
7. [ ] **GP Checkpoint:** Exemplar variance > 0 po 10 iteracích ✓

8. [ ] **Full BOLT run:** S opravenými komponentami (3 runy)

### Short-term (Weeks 2-3)
9. [ ] **BOLT > LIPO:** Ověřit statistickou významnost (p<0.05)
10. [ ] BBH dataset setup + evaluator
11. [ ] BBH experimenty (všechny metody)

### Medium-term (Weeks 4-5)
12. [ ] **Ablation A-E:** Implementace všech 5 variant
13. [ ] **Ablation runs:** 5 runů per variant
14. [ ] **Statistical tests:** t-test, Cohen's d

### Long-term (Weeks 6-8)
15. [ ] Visualization scripts (UMAP, sensitivity heatmap)
16. [ ] Paper draft (LaTeX)
17. [ ] Figure generation
18. [ ] Internal review
19. [ ] Final submission prep

---

## Metriky Úspěchu

### Phase 1 Checkpoints (Coordinate Descent)
| Checkpoint | Metrika | Cíl | Status |
|------------|---------|-----|--------|
| VAE Quality | Retrieval Accuracy @ 8 | ≥ 0.85 | ❌ Pending |
| Scorer Quality | NDCG@8 | ≥ 0.7 | ❌ Pending |
| GP Exploration | Exemplar variance after 10 iter | > 0 | ❌ Pending |
| Component Integration | All checkpoints passed | 3/3 | ❌ Pending |

### Final Milestones
| Milestone | Kritérium | Status |
|-----------|-----------|--------|
| BOLT > LIPO on GSM8K | 91.5% vs 90.5% (p<0.05) | ❌ Pending |
| BOLT > LIPO on BBH | +3% průměr | ❌ Pending |
| Ablation hierarchy | A < B < C < D < E confirmed | ❌ Pending |
| Sensitivity heatmap | Shows "islands" not "stripes" | ❌ Pending |
| **Smoothness verified** | Lipschitz < 10, hladké kontury | ❌ Pending |
| All ablations complete | 5 variant × 5 runs | ❌ Pending |
| Visualization ready | 6+ figures (UMAP, heatmap, smoothness, conv.) | ❌ Pending |
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

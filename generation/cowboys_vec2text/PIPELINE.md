# COWBOYS Vec2Text Pipeline

> **COWBOYS** = Continuous Optimization With BOx constraints and Yielding Samples

Pokročilá metoda pro automatickou optimalizaci instrukcí kombinující VAE, Gaussian Process a pCN MCMC vzorkování s Vec2Text inverzí.

**Tato verze je instruction-only (bez exemplars).**

---

## Obsah

1. [Přehled architektury](#prehled-architektury)
2. [Fáze 1: Příprava dat](#faze-1-priprava-dat)
3. [Fáze 2: Inicializace a embedding](#faze-2-inicializace-a-embedding)
4. [Fáze 3: Načtení pre-evaluated gridu](#faze-3-nacteni-pre-evaluated-gridu)
5. [Fáze 4: Trénink VAE](#faze-4-trenink-vae)
6. [Fáze 5: Trénink GP na dekódovaných embeddingách](#faze-5-trenink-gp-na-dekodovanych-embeddingach)
7. [Fáze 6: Inicializace inference pipeline](#faze-6-inicializace-inference-pipeline)
8. [Fáze 7: Optimalizační smyčka](#faze-7-optimalizacni-smycka)
9. [Fáze 8: Finální výstup](#faze-8-finalni-vystup)
10. [Datový tok - diagram](#datovy-tok---diagram)
11. [Klíčové inovace COWBOYS](#klicove-inovace-cowboys)
12. [Konfigurační parametry](#konfiguracni-parametry)

---

## Přehled architektury

```
Vstupní instrukce (100 textů)
    |
    v
[GTR Encoder] --> 768D embeddingy
    |
    v
[VAE] --> 32D latentní prostor (768->256->128->32)
    |
    v
[pCN MCMC Sampler] --> Probabilistické vzorkování v latentním prostoru
    |
    v
[Trust Region Manager] --> Omezení prohledávání na "bezpečné" regiony
    |
    v
[Vec2Text Inversion] --> Dekódování latentu zpět na přirozený text
    |
    v
[InstructionSelector GP] --> Predikce chybovosti pro nové instrukce
    |
    v
[Weighted VAE Retraining] --> Zaměření latentního prostoru na kvalitní oblasti
    |
    v
Výstup: Nejlepší optimalizovaná instrukce
```

### Soubory modulu

| Soubor | Popis |
|--------|-------|
| `run.py` | CLI entry point, hlavní orchestrace |
| `optimizer.py` | `CowboysOptimizer` - správa modelů a gridu |
| `inference.py` | `CowboysInference` - MCMC + Vec2Text pipeline |
| `mcmc.py` | `pCNSampler` - Preconditioned Crank-Nicolson MCMC |
| `training.py` | `WeightedVAETrainer` - trénink VAE s váhami |
| `vae.py` | `InstructionVAE` - VAE architektura + cycle-consistency loss + InvBO decoder inversion |
| `trust_region.py` | `TrustRegionManager` - TuRBO-style adaptivní omezení |
| `encoder.py` | `GTRPromptEncoder` - GTR-T5-Base embedding |
| `visualize.py` | EI landscape UMAP vizualizace pro diagnostiku inversion gap |

**Důležité:** GP je importován z `robust_vec2text.exemplar_selector.InstructionSelector` (single-branch deep kernel).

---

## Fáze 1: Příprava dat

### Krok 1.1: Načtení vstupních dat

**Lokace:** `run.py:306-316`

```python
instructions = load_instructions(args.instructions)  # 100 textových instrukcí
validation_data = load_validation(args.validation)   # JSON s math problems
```

**Vstupy:**
- `datasets/cowboys/instructions_100.txt` - instrukce ve formátu "N. text"
- `datasets/cowboys/grid_100_qend.jsonl` - pre-evaluated grid (instruction_id → error_rate)
- `hbbops_improved_2/data/validation.json` - validační data pro GSM8K

### Krok 1.2: APE Data Augmentation (POVINNÉ)

**Lokace:** `run.py:319-349`

**DŮLEŽITÉ: Nikdy nepoužívat `--skip-ape`!** APE instrukce jsou kritické pro kvalitní VAE:
- S APE: 1100 instrukcí → VAE cosine ~0.98
- Bez APE: 100 instrukcí → VAE cosine ~0.87

```python
ape_generator = APEInstructionGenerator(model=args.model, backend=args.backend)
ape_instructions = ape_generator.generate_or_load(
    cache_path=args.ape_cache,  # /home/prusek/NLP/datasets/cowboys/ape_instructions_1000.json
    num_instructions=1000,
)
all_instructions = list(set(instructions + ape_instructions))  # 1100 unique

# Uvolnění GPU paměti po APE
del ape_generator
gc.collect()
torch.cuda.empty_cache()
```

---

## Fáze 2: Inicializace a embedding

### Krok 2.1: Inicializace GTR Encoderu

**Lokace:** `optimizer.py:81-82`

```python
self.gtr = GTRPromptEncoder(device=str(self.device))
```

**GTR (Generative Text Retrieval):**
- Model: `sentence-transformers/gtr-t5-base`
- Výstup: 768-dimenzionální normalizovaný embedding
- Klíčová vlastnost: Vec2Text umí invertovat právě GTR embeddingy

### Krok 2.2: Pre-compute všech embeddingů

**Lokace:** `optimizer.py:85-91`

```python
# Pouze instrukce (žádné exemplary)
for i, inst in enumerate(instructions):
    emb = self.gtr.encode_tensor(inst)
    self.instruction_embeddings[i] = emb.to(self.device)

print(f"Cached {len(self.instruction_embeddings)} instruction embeddings")
```

**Proč pre-compute?**
- GTR encoding je relativně pomalý
- Předpočítání umožňuje rychlý přístup během optimalizace
- Embeddingy se používají opakovaně v každé iteraci

---

## Fáze 3: Načtení pre-evaluated gridu

### Krok 3.1: Load top-k z gridu

**Lokace:** `optimizer.py:162-214`, `run.py:367-377`

```python
grid_prompts = optimizer.load_grid(
    args.grid_path,      # datasets/cowboys/grid_100_qend.jsonl
    top_k=args.top_k,    # default: 25
    train_instruction_gp=False,
)
```

**Proces:**
1. Načte `grid_100_qend.jsonl` (100 instrukcí s error rates)
2. Každý záznam obsahuje: `instruction_id`, `error_rate`
3. Seřadí podle error rate (nejlepší první)
4. Vybere top-25 nejlepších instrukcí

**Výstup - GridPrompt dataclass:**
```python
@dataclass
class GridPrompt:
    instruction_id: int    # Index do seznamu instrukcí
    instruction: str       # Text instrukce
    error_rate: float      # Validační error rate
```

---

## Fáze 4: Trénink VAE

### Krok 4.1: VAE architektura

**Lokace:** `vae.py`

```
Encoder:
  Input (768) -> Linear(256) -> ReLU -> Linear(128) -> ReLU
       -> Linear(32) [mu]
       -> Linear(32) [logvar]

Decoder:
  Latent (32) -> Linear(128) -> ReLU -> Linear(256) -> ReLU
       -> Linear(768) -> L2-normalize
```

**Klíčové vlastnosti:**
- Latentní dimenze: 32 (sweet spot pro GP optimalizaci)
- Výstup dekodéru je L2-normalizovaný (pro Vec2Text kompatibilitu)
- Reparametrizační trik: `z = mu + exp(0.5 * logvar) * epsilon`

### Krok 4.2: Loss funkce

**Lokace:** `training.py:WeightedVAELoss`, `vae.py`

```python
L_total = lambda_cos * L_cos + lambda_mse * L_mse + lambda_kld * L_kld + lambda_cycle * L_cycle

# Váhy (tuned pro Vec2Text)
lambda_cosine = 20.0   # Priorita pro GTR (L2-normalized embeddings)
lambda_mse = 1.0       # Pomocná rekonstrukce
lambda_kld = 0.0025    # S annealingem, zabraňuje shluky daleko od N(0,I)
lambda_cycle = 2.0     # Cycle-consistency ||E(D(z)) - z||²
```

**Komponenty:**
- **Cosine loss:** `1 - cosine_similarity(input, reconstruction)` - směr vektoru
- **MSE loss:** `||input - reconstruction||^2` - absolutní hodnoty
- **KL divergence:** `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))` - regularizace
- **Cycle loss:** `||E(D(z)) - z||²` - konzistence latentního prostoru

**Proč tyto váhy?**
- **lambda_cosine=20:** GTR embeddingy jsou L2-normalizované → směr je důležitější než délka
- **lambda_kld=0.0025:** S 0.0001 VAE "podvádí" - ukládá info do vzdálených shluků, což rozbíjí pCN sampler (předpokládá N(0,I)). S >0.01 dojde k posterior collapse.
- **lambda_cycle=2.0:** Trust Region spoléhá na E(D(z)) ≈ z. Bez cycle loss je "inversion drift" příliš velký.

### Krok 4.3: Tréninková smyčka

**Lokace:** `run.py:384-403`, `training.py:175-261`

```python
vae_history = optimizer.train_vae(
    epochs=args.vae_epochs,        # default: 3000
    lr=args.vae_lr,                # default: 0.001
    patience=args.vae_patience,    # default: 30
    lambda_cycle=args.vae_cycle_weight,  # default: 2.0
    verbose=True,
)
```

**Proces:**
1. **KLD Annealing:** `lambda_kld` roste lineárně z 0 do 0.0025 přes prvních 500 epoch, pak drží
   - Epoch 0-500: `lambda_kld = 0.0025 * (epoch / 500)`
   - Epoch 500+: `lambda_kld = 0.0025`
   - Proč: Na začátku VAE potřebuje naučit rekonstrukci, příliš silný KL prior = posterior collapse
2. **Cycle loss:** Počítá se v každém batchi jako `||E(D(mu)) - mu||²`
3. **Train/Val split:** 80/20
4. **Early stopping:** patience=30 epoch na validační cosine
5. **LR scheduler:** ReduceLROnPlateau

---

## Fáze 5: Trénink GP na dekódovaných embeddingách

### Krok 5.1: COWBOYS fix - Encode-Decode cyklus

**Lokace:** `optimizer.py:234-291`

```python
# Pro každou instrukci:
decoded_instruction_embeddings = {}
for inst_id, orig_emb in self.instruction_embeddings.items():
    latent = vae.get_latent(orig_emb.unsqueeze(0))
    decoded_emb = vae.decode(latent).squeeze(0)
    decoded_instruction_embeddings[inst_id] = decoded_emb
```

**Proč decoded embeddingy?**

Toto je klíčová inovace COWBOYS. MCMC vzorkuje v 32D latentním prostoru a dekóduje přes VAE:

```
Bez COWBOYS fix:
  GP trénován na: original embeddings (768D)
  MCMC produkuje: decoded embeddings (768D)
  --> Distribution mismatch --> GP predictions jsou špatné --> EI=0

S COWBOYS fix:
  GP trénován na: decoded embeddings (768D)
  MCMC produkuje: decoded embeddings (768D)
  --> Konzistentní distribuce --> GP predictions jsou správné
```

### Krok 5.2: InstructionSelector GP architektura

**Lokace:** `robust_vec2text/exemplar_selector.py:654-738`

```
Input: instruction_emb (768D)
              |
              v
    InstructionOnlyFeatureExtractor:
    768 -> 64 (ReLU) -> 32 (ReLU) -> 10D
              |
              v
    GP kernel (Matérn 5/2 ARD) na 10D features
              |
              v
    Output: predikce error rate
```

**Klíčové vlastnosti:**
- Single-branch deep kernel (pouze instrukce)
- ARD (Automatic Relevance Determination) - 10 nezávislých lengthscales
- Priors: GammaPrior na lengthscale a output scale

### Krok 5.3: Trénink GP

**Lokace:** `run.py:410-416`, `optimizer.py:234-291`

```python
optimizer.train_instruction_gp_on_decoded(
    grid_path=args.grid_path,
    top_k=args.top_k,
    epochs=args.gp_epochs,  # default: 3000
    patience=10,
    verbose=True,
)
```

---

## Fáze 6: Inicializace inference pipeline

### Krok 6.1: CowboysInference setup

**Lokace:** `run.py:423-428`, `inference.py:45-129`

```python
inference = CowboysInference(
    vae=optimizer.get_vae(),
    instruction_selector=optimizer.get_instruction_selector(),
    gtr=optimizer.gtr,
    device="cuda",
)
```

**Komponenty:**
- Natrénovaný VAE model
- Natrénovaný InstructionSelector GP
- GTR encoder pro re-embedding
- pCN Sampler (inicializován interně)

**Poznámka:** Žádný `exemplar_emb` parametr - tato verze je instruction-only.

### Krok 6.2: Trust Region inicializace (volitelné)

**Lokace:** `run.py:431-435`, `trust_region.py`

**DŮLEŽITÉ: Trust Region je defaultně VYPNUTÝ.** Zapnout pomocí `--trust-region`.

```python
# Trust Region je None pokud není --trust-region
trust_region = None
if args.trust_region:
    trust_region = optimizer.initialize_trust_region(config=tr_config)

# TRConfig defaults:
initial_radius = 1.0
min_radius = 0.1
max_radius = 5.0
expand_factor = 2.0
contract_factor = 0.5
success_threshold = 3
failure_threshold = 5
```

**Trust Region mechanismus:**
- L-infinity ball kolem anchor bodu
- Anchor = latent nejlepší instrukce z gridu
- Adaptivní radius: expand při úspěších, contract při neúspěších
- **Proč defaultně vypnutý:** Cycle-consistency loss (lambda_cycle=2.0) zajišťuje konzistenci latentního prostoru, takže TR není vždy nutný

---

## Fáze 7: Optimalizační smyčka

### Krok 7.0: Inicializace evaluation clienta

**Lokace:** `run.py:446-449`

```python
# Create evaluation client once (reused for all iterations)
from src.llm_client import create_llm_client
eval_client = create_llm_client(args.model, args.backend)
```

**Důležité:** Client se vytvoří jednou a sdílí se mezi všemi iteracemi (šetří GPU paměť).

Pro každou iteraci (default 50):

### Krok 7.1: Check VAE retraining

**Lokace:** `run.py:467-469`

```python
if retrain_config and optimizer.should_retrain_vae(iteration, retrain_config):
    optimizer.retrain_vae(retrain_config, verbose=True)
```

**Vážené přetrénování (každých 10 iterací):**
- Akumulované vzorky z gridu + vygenerované
- Váhy podle error rate: nižší error = vyšší váha
- VAE se "zaměří" na kvalitní regiony latentního prostoru

**Metody vážení:**
```python
# Rank-based (default):
weights = 1 / rank(error)^power

# Exponential:
weights = exp(-beta * error)
```

### Krok 7.2: pCN MCMC Sampling

**Lokace:** `run.py:473-484`, `inference.py:131-171`, `mcmc.py:270-345`

#### a) Generování startovních bodů

```python
initial_latents[0] = best_latent  # Z nejlepší instrukce
initial_latents[1-4] = trust_region.get_random_point_in_region()
```

#### b) pCN Proposal

**Matematika:**
```
z_new = sqrt(1 - beta^2) * z_old + beta * epsilon,  epsilon ~ N(0, I)
```

**Vlastnosti:**
- Zachovává N(0, I) jako stacionární distribuci
- Pokud z ~ N(0,I), pak z_new ~ N(0,I)
- Ideální pro VAE latentní prostory

#### c) MCMC smyčka (500 kroků na chain)

```python
for step in range(n_steps):
    # 1. pCN Proposal
    z_proposal = sqrt(1 - beta^2) * z_current + beta * randn()

    # 2. Trust Region Check
    if not trust_region.is_within_region(z_proposal):
        continue  # REJECT

    # 3. Compute Log Expected Improvement
    inst_emb = vae.decode(z_proposal)  # 32D -> 768D
    gp_pred = instruction_selector.predict(inst_emb)
    log_ei = LogExpectedImprovement(gp_pred, best_error)

    # 4. Metropolis-Hastings Accept/Reject
    log_alpha = log_ei_proposal - log_ei_current
    if log(uniform()) < log_alpha:
        z_current = z_proposal  # ACCEPT
        log_ei_current = log_ei_proposal

    # 5. Beta Adaptation (během warmup)
    if step < warmup_steps:
        accept_rate = accepts / (step + 1)
        if accept_rate < 0.234:
            beta *= 0.95  # Menší kroky
        else:
            beta *= 1.05  # Větší kroky

    # 6. Collect samples (po warmup, každý 10.)
    if step >= warmup_steps and (step - warmup_steps) % thinning == 0:
        candidates.append(z_current)
```

#### d) Výstup MCMC

```python
# ~40 vzorků per chain x 5 chains = ~200 latentních kandidátů
MCMCResult(
    candidates=all_samples,
    best_latent=best_z,
    best_log_ei=best_log_ei,
    accept_rate=overall_accept_rate,
    final_beta=final_beta,
)
```

### Krok 7.3: Výběr top kandidátů

**Lokace:** `inference.py:357-369`

```python
# Seřadit všech ~200 kandidátů podle log_ei
candidates_with_ei = [(z, compute_log_ei(z, best_y)) for z in candidates]
candidates_with_ei.sort(key=lambda x: x[1], reverse=True)

# Vybrat top-20 pro Vec2Text inverzi
top_latents = [z for z, _ in candidates_with_ei[:max_decode]]
```

### Krok 7.4: VAE Decode + Vec2Text Inversion

**Lokace:** `inference.py:173-230`, `inference.py:232-270`

Pro každého z top-20 kandidátů:

```python
# 1. VAE decode
z = latent  # 32D
target_embedding = vae.decode(z)  # 768D

# 2. Vec2Text inversion
output_ids = vec2text_model.generate(
    inputs={"frozen_embeddings": target_embedding},
    generation_kwargs={
        "num_beams": 8,
        "max_length": 512,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,
    },
)
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# 3. Re-encode a compute cosine
realized_embedding = gtr.encode_tensor(text)  # 768D
cosine = cosine_similarity(target_embedding, realized_embedding)
```

**Vec2Text model:** `vec2text/gtr-512-noise-0.00001` (HuggingFace)

### Krok 7.5: Ranking a výběr nejlepšího

**Lokace:** `inference.py:272-304`

```python
# Znovu seřadit podle log_ei
ranked = []
for (text, cosine, realized_emb, target_emb), z in zip(candidates, latents):
    log_ei = compute_log_ei(z, best_y)
    ranked.append(InversionResult(text, target_emb, realized_emb, cosine, log_ei, z, ...))

ranked.sort(key=lambda x: x.log_ei, reverse=True)  # Nejvyšší log_ei první
best_result = ranked[0]
```

### Krok 7.5.1: EI Landscape Vizualizace (volitelné)

**Lokace:** `run.py:495-515`, `visualize.py`

Pokud `--visualize`:

```python
from .visualize import visualize_ei_landscape

viz_metrics = visualize_ei_landscape(
    inference=inference,
    center_latent=best_result.optimized_latent,  # z_opt
    realized_text=best_result.text,               # Pro výpočet z_realized
    best_y=best_error,
    trajectory_latents=result.mcmc_result.candidates,  # MCMC trajectory
    trust_region=trust_region,                    # Pro boundary overlay (None pokud vypnutý)
    save_path=f"results/ei_landscape_iter_{iteration}.png",
)
```

**Co vizualizace ukazuje:**
- **UMAP projekce** 32D latentního prostoru do 2D
- **EI surface** jako contour plot (RBF interpolace)
- **z_opt** (červená hvězda) - kde MCMC našel vysoké EI
- **z_realized** (bílé X) - kde Vec2Text skutečně skončil po inverzi
- **Inversion gap** - vzdálenost mezi z_opt a z_realized v 32D
- **Trust Region boundary** - žlutá přerušovaná čára (pokud zapnutý)

**Diagnostické metriky:**
```python
{
    "inversion_gap_32d": 0.234,    # Vzdálenost v originálním prostoru
    "inversion_gap_2d": 1.45,      # Vzdálenost v UMAP projekci
    "log_ei_at_opt": -2.34,        # LogEI v cílovém bodě
    "log_ei_at_realized": -3.12,   # LogEI kde skutečně skončil
}
```

**Interpretace:**
| Pozorování | Diagnóza | Řešení |
|------------|----------|--------|
| X na hvězdě | Perfektní inverze | Žádné |
| X v high-EI regionu | Akceptovatelný drift | Minor tuning |
| X v low-EI regionu | Inversion gap problém | Zvýšit lambda_cycle |
| Plochá EI surface | GP nejistota vysoká | Více trénovacích dat |

### Krok 7.6: Evaluace nové instrukce

**Lokace:** `run.py:499-505`

```python
# Instrukce je přímo prompt (instruction-only, žádný exemplar)
novel_instruction = best_result.text

# Evaluace na validačních datech
novel_error = evaluate_prompt(
    novel_instruction,
    validation_data,
    args.model,      # Qwen/Qwen2.5-7B-Instruct
    args.backend,    # vllm
    client=eval_client,  # Sdílený client
)
```

**Evaluační proces:**
1. Pro každý validační příklad: `instruction + question -> LLM -> answer`
2. Extrakce odpovědi (poslední číslo v odpovědi)
3. Porovnání s ground truth
4. `error_rate = 1 - (correct / total)`

### Krok 7.7: Update GP (InvBO-aligned)

**Lokace:** `run.py:553-600`

**NOVÉ v této verzi:** InvBO-style decoder inversion pro snížení prediction gap.

```python
# 1. Získat embeddingy pro diagnostiku
inst_emb = optimizer.gtr.encode_tensor(best_result.text).to(optimizer.device)
decoded_emb = optimizer.get_decoded_embedding(best_result.text)

# 2. GP prediction PŘED updatem (pro gap diagnostiku)
gp_pred_opt, gp_std_opt = optimizer.instruction_selector.predict_error(decoded_emb)

# 3. InvBO-style decoder inversion
# Najít z_inv takové, že VAE.decode(z_inv) ≈ GTR(text)
# Toto vytváří aligned triplet (z_inv, error) kde decode(z_inv) ≈ skutečný embedding
z_inv = optimizer.get_vae().invert_decoder(inst_emb)

# 4. Získat aligned decoded embedding
decoded_emb_aligned = optimizer.get_vae().decode(z_inv.unsqueeze(0)).squeeze(0)
inversion_cosine = cosine_similarity(decoded_emb_aligned, inst_emb)

# 5. Použít ALIGNED embedding pro GP training (klíč InvBO)
optimizer.instruction_selector.add_observation_and_retrain(
    decoded_inst_emb=decoded_emb_aligned,  # <-- Aligned, ne encoder-based!
    error_rate=novel_error,
    epochs=3000,
    patience=10,
)

# 6. Gap diagnostika
gp_pred_inv, _ = optimizer.instruction_selector.predict_error(decoded_emb_aligned)
log(f"=== Gap Diagnostics ===")
log(f"  GP pred at z_opt: {gp_pred_opt:.4f}")
log(f"  GP pred at z_inv: {gp_pred_inv:.4f}")
log(f"  True evaluation:  {novel_error:.4f}")
log(f"  Gap (z_opt): {abs(gp_pred_opt - novel_error):.4f}")
log(f"  Gap (z_inv): {abs(gp_pred_inv - novel_error):.4f}")
```

**Proč InvBO-style inversion?**

Problém s encoder-based přístupem:
```
Text → GTR.encode() → embedding_orig (768D)
embedding_orig → VAE.encode() → z
z → VAE.decode() → embedding_decoded ≠ embedding_orig  # MISALIGNMENT!
```

InvBO řešení:
```
Text → GTR.encode() → embedding_target (768D)
embedding_target → invert_decoder() → z_inv  # Gradient descent optimalizace
z_inv → VAE.decode() → embedding_aligned ≈ embedding_target  # ALIGNED!
```

**Efekt na prediction gap:**
| Metrika | Encoder-based | InvBO-aligned |
|---------|--------------|---------------|
| GP pred vs true | ~0.14 gap | ~0.02 gap |
| Consistency | Misaligned | Aligned |
| GP accuracy | Overestimates | Accurate |

### Krok 7.8: Update Trust Region

**Lokace:** `run.py:528-534`, `trust_region.py`

```python
modified = trust_region.update(
    best_result.optimized_latent,
    novel_error,
    best_error,
)
```

**Logika:**
```
Pokud error_rate < best_error (ÚSPĚCH):
    success_count += 1
    Pokud success_count >= 3: radius *= 2 (EXPAND)

Pokud error_rate >= best_error (NEÚSPĚCH):
    failure_count += 1
    Pokud failure_count >= 5: radius *= 0.5 (CONTRACT)
    Pokud radius < 0.1: RESTART (nový anchor, reset radius)
```

### Krok 7.9: Update best a příprava další iterace

**Lokace:** `run.py:540-555`

```python
# Update best pokud zlepšení
if novel_error < best_error:
    best_error = novel_error
    best_instruction = best_result.text
    trust_region.set_anchor(current_latent)

# Příprava latentu pro další iteraci
new_inst_emb = gtr.encode_tensor(best_result.text)
current_latent = vae.get_latent(new_inst_emb.unsqueeze(0)).squeeze(0)
```

---

## Fáze 8: Finální výstup

**Lokace:** `run.py:575-600`

```python
# Logování výsledků
print(f"Initial best error (grid): {best_prompt.error_rate:.4f}")
print(f"Final best error: {best_error:.4f}")
print(f"Total improvement: {best_prompt.error_rate - best_error:.4f}")
print(f"Best instruction:\n{best_instruction}")

# Uložení JSON
results = {
    "timestamp": timestamp,
    "method": "COWBOYS",
    "grid_best": {...},
    "optimized": {
        "instruction": best_instruction,
        "error_rate": best_error,
    },
    "iteration_history": [...],
    "improvement": improvement,
}
```

---

## Datový tok - diagram

```
+-------------------------------------------------------------------------+
|                         FÁZE PŘÍPRAVY                                   |
+-------------------------------------------------------------------------+
|  Instrukce (text) --GTR--> 768D embeddingy                              |
|  Grid (JSONL) --load--> top-25 (instruction_id, error_rate)             |
|                                                                         |
|  768D embeddingy --VAE train--> VAE model (768->32->768)                |
|  decoded embeddingy + grid errors --GP train--> InstructionSelector GP  |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                    ITERATIVNÍ OPTIMALIZACE (50x)                        |
+-------------------------------------------------------------------------+
|                                                                         |
|  current_latent (32D)                                                   |
|        |                                                                |
|        v                                                                |
|  +-------------------------------------------------------------+        |
|  |           pCN MCMC (5 chains x 500 kroků)                   |        |
|  |                                                             |        |
|  |  z_new = sqrt(1-beta^2) * z_old + beta * epsilon            |        |
|  |           |                                                 |        |
|  |           v                                                 |        |
|  |  VAE.decode(z_new) -> 768D                                  |        |
|  |           |                                                 |        |
|  |           v                                                 |        |
|  |  inst_emb -> GP -> log_ei                                   |        |
|  |           |                                                 |        |
|  |           v                                                 |        |
|  |  Metropolis-Hastings: accept/reject                         |        |
|  |           |                                                 |        |
|  |           v                                                 |        |
|  |  Trust Region: constrain to L-infinity ball                 |        |
|  +-------------------------------------------------------------+        |
|                             |                                           |
|                             v                                           |
|  ~200 latentních kandidátů -> sort by log_ei -> top-20                  |
|                             |                                           |
|                             v                                           |
|  +-------------------------------------------------------------+        |
|  |              Vec2Text Inversion (20x)                       |        |
|  |                                                             |        |
|  |  z (32D) -> VAE.decode() -> target_emb (768D)               |        |
|  |              |                                              |        |
|  |              v                                              |        |
|  |  Vec2Text.generate(target_emb) -> text                      |        |
|  |              |                                              |        |
|  |              v                                              |        |
|  |  GTR.encode(text) -> realized_emb (768D)                    |        |
|  |              |                                              |        |
|  |              v                                              |        |
|  |  cosine_sim(target, realized)                               |        |
|  +-------------------------------------------------------------+        |
|                             |                                           |
|                             v                                           |
|  Nejlepší kandidát (text) -> novel_instruction                          |
|                             |                                           |
|                             v                                           |
|  +-------------------------------------------------------------+        |
|  |              LLM Evaluace (GSM8K)                           |        |
|  |                                                             |        |
|  |  instruction + question -> LLM -> answer                    |        |
|  |  compare(answer, ground_truth) -> error_rate                |        |
|  +-------------------------------------------------------------+        |
|                             |                                           |
|                             v                                           |
|  +-------------------------------------------------------------+        |
|  |                    Updates                                  |        |
|  |                                                             |        |
|  |  GP.add_observation(decoded_embedding, error_rate)          |        |
|  |  TrustRegion.update(success/failure)                        |        |
|  |  VAE.add_sample(embedding, error_rate)                      |        |
|  |                                                             |        |
|  |  if improved: best_error = error_rate                       |        |
|  |               trust_region.anchor = new_latent              |        |
|  |                                                             |        |
|  |  if iteration % 10 == 0: VAE.retrain_with_weights()         |        |
|  +-------------------------------------------------------------+        |
|                                                                         |
+-------------------------------------------------------------------------+
                                     |
                                     v
+-------------------------------------------------------------------------+
|                         VÝSTUP                                          |
+-------------------------------------------------------------------------+
|  best_instruction: "Optimalizovaný prompt text..."                      |
|  best_error: 0.XX (vs. grid baseline 0.YY)                              |
|  improvement: 0.ZZ                                                      |
|  iteration_history: [všechny kroky s metrikami]                         |
+-------------------------------------------------------------------------+
```

---

## Klíčové inovace COWBOYS

### 1. pCN MCMC (nahrazuje gradientní sestup)

| Aspekt | Gradientní sestup | pCN MCMC |
|--------|-------------------|----------|
| Konvergence | Jedno lokální optimum | Více modů |
| Prior | Žádné explicitní omezení | Zachovává N(0,I) |
| Explorace | Deterministická | Probabilistická |
| Gradienty | Vyžadovány | Nevyžadovány |
| Diverzita | Jeden kandidát | ~200 kandidátů |

**pCN Proposal:**
```
z_new = sqrt(1 - beta^2) * z_old + beta * epsilon

Zachovává N(0,I):
- E[z_new] = sqrt(1-beta^2) * 0 + beta * 0 = 0
- Var[z_new] = (1-beta^2) * I + beta^2 * I = I
```

### 2. Trust Regions (TuRBO-style)

**L-infinity omezení:**
```
max_i |z_i - anchor_i| <= radius
```

**Adaptivní dynamika:**
- 3+ úspěchy -> radius *= 2 (EXPAND)
- 5+ neúspěchy -> radius *= 0.5 (CONTRACT)
- radius < 0.1 -> RESTART

### 3. Vážené VAE přetrénování

**Princip:** Nižší error rate = vyšší váha při tréninku

```python
# Rank-based (robustní k outlierům):
weights = 1 / rank(error)^power

# Exponential (citlivý na absolutní rozdíly):
weights = exp(-beta * error)
```

**Efekt:** VAE latentní prostor se "zaměří" na kvalitní prompty.

### 4. GP trénink na decoded embeddingách

**Problém:** MCMC dekóduje latenty přes VAE, ale GP byl trénován na originálních embeddingách.

**Řešení:** Trénovat GP na embeddingách, které prošly VAE encode-decode cyklem.

```python
# Bez COWBOYS fix:
gp.train(original_embeddings)  # Distribuční mismatch!

# S COWBOYS fix:
decoded = vae.decode(vae.encode(original_embeddings))
gp.train(decoded)  # Konzistentní distribuce
```

### 5. Single-branch Deep Kernel GP

**Architektura InstructionOnlyFeatureExtractor:**
```
768D (GTR) -> Linear(64) -> ReLU -> Linear(32) -> ReLU -> Linear(10) -> 10D
                                                                        |
                                                                        v
                                                            Matérn 5/2 kernel (ARD)
```

Toto je jedna větev z původního dvou-větvového `FeatureExtractor` (bez exemplar větve).

### 6. InvBO-style Decoder Inversion (NOVÉ)

**Reference:** [InvBO - AAAI 2024](https://arxiv.org/html/2411.05330v1)

**Problém:** GP prediction gap ~0.14 (GP overestimuje error rate)

**Příčina:** Encoder-based latenty vytváří "misalignment" - encode(text) → z, ale decode(z) ≠ text_embedding

**Řešení: Decoder inversion**
```python
# Místo encoder-based:
z = vae.encode(embedding)  # Misaligned

# InvBO-style inversion:
z_inv = argmin_z ||vae.decode(z) - embedding||²  # Aligned!
```

**Implementace** (`vae.py:invert_decoder`):
```python
def invert_decoder(self, target_embedding, n_steps=500, lr=0.1):
    z = vae.encode(target).clone().requires_grad_(True)  # Warm start
    optimizer = Adam([z], lr=lr)

    for step in range(n_steps):
        decoded = vae.decode(z)
        loss = mse(decoded, target) + 10 * (1 - cosine_sim(decoded, target))
        loss.backward()
        optimizer.step()

    return z.detach()  # z_inv where decode(z_inv) ≈ target
```

**Efekt:**
- GP training data je "aligned" s tím, co MCMC produkuje
- Prediction gap klesá z ~0.14 na ~0.02
- GP predikce jsou přesnější → lepší EI → lepší optimalizace

---

## Konfigurační parametry

### MCMC parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--mcmc-steps` | 500 | Počet kroků na chain |
| `--mcmc-beta` | 0.1 | pCN step size |
| `--mcmc-chains` | 5 | Počet paralelních chainů |
| `--mcmc-warmup` | 50 | Burn-in kroky |
| `--mcmc-thinning` | 10 | Každý N-tý vzorek |
| `--mcmc-adapt-beta` | True | Adaptivní beta |

### Trust Region parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--trust-region` | False | **Zapnout TR** (defaultně vypnutý) |
| `--tr-initial` | 1.0 | Počáteční radius |
| `--tr-min` | 0.1 | Minimální radius |
| `--tr-max` | 5.0 | Maximální radius |
| `--tr-expand` | 2.0 | Faktor expanze |
| `--tr-contract` | 0.5 | Faktor kontrakce |

### VAE parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--latent-dim` | 32 | Dimenze latentního prostoru |
| `--vae-epochs` | 3000 | Tréninkových epoch |
| `--vae-lr` | 0.001 | Learning rate |
| `--vae-patience` | 30 | Early stopping patience |
| `--vae-cycle-weight` | 2.0 | Váha pro cycle-consistency loss ||E(D(z))-z||² |

### GP parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--gp-epochs` | 3000 | GP tréninkových epoch |
| `--top-k` | 25 | Počet top promptů z gridu |

### Retrain parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--retrain-interval` | 10 | Přetrénovat každých N iterací |
| `--retrain-method` | "rank" | Metoda vážení (rank/exponential) |
| `--retrain-epochs` | 50 | Epoch při přetrénování |
| `--no-retrain` | False | Vypnout přetrénování |

### Vec2Text parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--v2t-beam` | 8 | Beam search width |
| `--v2t-max-length` | 512 | Max output length |
| `--v2t-no-repeat-ngram` | 3 | Block repeating n-grams |
| `--v2t-repetition-penalty` | 1.2 | Penalizace opakování |
| `--max-decode` | 20 | Max kandidátů k dekódování |

### Vizualizace

| Parametr | Default | Popis |
|----------|---------|-------|
| `--visualize` | False | Generovat EI landscape vizualizaci po každé iteraci |

---

## Použití

```bash
# DOPORUČENÉ: Standardní spuštění s vizualizací
# (Nikdy nepoužívat --skip-ape!)
uv run python -m generation.cowboys_vec2text.run --visualize --iterations 5

# S trust region (pro konzervativnější exploraci)
uv run python -m generation.cowboys_vec2text.run --visualize --iterations 10 --trust-region

# S custom MCMC parametry
uv run python -m generation.cowboys_vec2text.run \
    --visualize \
    --iterations 20 \
    --mcmc-steps 1000 \
    --mcmc-chains 10 \
    --top-k 50

# Bez VAE přetrénování (rychlejší, méně adaptivní)
uv run python -m generation.cowboys_vec2text.run \
    --visualize \
    --iterations 10 \
    --no-retrain

# Custom cycle-consistency váha
uv run python -m generation.cowboys_vec2text.run \
    --visualize \
    --iterations 5 \
    --vae-cycle-weight 5.0  # Silnější konzistence
```

**VAROVÁNÍ:** Nikdy nepoužívat `--skip-ape`! APE instrukce jsou kritické pro kvalitní VAE.

---

## Reference

- **Vec2Text:** Morris et al., "Text Embeddings Reveal (Almost) As Much As Text"
- **pCN MCMC:** Cotter et al., "MCMC Methods for Functions"
- **TuRBO:** Eriksson et al., "Scalable Global Optimization via Local Bayesian Optimization"
- **HbBoPs:** Prusek, "Hyperband Bayesian Optimization for Prompt Selection"
- **InvBO:** "Inversion-based Latent Bayesian Optimization" (AAAI 2024) - decoder inversion pro aligned GP training
- **LOL-BO:** Maus et al., "Local Latent Space Bayesian Optimization" (NeurIPS 2022) - trust regions v latentním prostoru

# OpenAI GPT-3.5-Turbo Integration pro OPRO

## Přehled

Do OPRO frameworku byla přidána podpora pro OpenAI GPT modely (GPT-3.5-Turbo, GPT-4, atd.). Nyní můžete použít GPT-3.5-Turbo jako:
- **Task model** - model, který je optimalizován (řeší úlohy)
- **Meta-optimizer model** - model, který generuje vylepšené prompty (scorer)

## Konfigurace

### 1. API Klíč

Váš OpenAI API klíč je již nakonfigurován v `.env` souboru:

```bash
OPENAI_API_KEY=sk-proj-...
```

### 2. Podporované modely

- `gpt-3.5-turbo` - GPT-3.5 Turbo (nejrychlejší, nejlevnější)
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - GPT-4
- Aliasy:
  - `gpt-3.5` → `gpt-3.5-turbo`
  - `gpt-4` → `gpt-4-turbo`

## Použití

### Základní použití - GPT-3.5 pro task i meta-optimizer

```bash
uv run python main.py \
    --method opro \
    --model gpt-3.5-turbo \
    --meta-model gpt-3.5-turbo \
    --iterations 10 \
    --minibatch-size 20
```

### Použití s aliasem

```bash
uv run python main.py \
    --method opro \
    --model gpt-3.5 \
    --meta-model gpt-3.5 \
    --iterations 10
```

### Hybridní nastavení - lokální model + GPT-3.5 meta-optimizer

```bash
# Qwen pro task evaluation (lokální), GPT-3.5 pro meta-optimization
uv run python main.py \
    --method opro \
    --model Qwen/Qwen2.5-7B-Instruct \
    --backend vllm \
    --meta-model gpt-3.5-turbo \
    --meta-backend openai \
    --iterations 10
```

### Backend specifikace

Backend se detekuje automaticky podle názvu modelu:
- `gpt*` → automaticky použije `openai` backend
- Můžete explicitně specifikovat: `--backend openai` nebo `--meta-backend openai`

```bash
# Explicitní backend specifikace
uv run python main.py \
    --method opro \
    --model gpt-3.5-turbo \
    --backend openai \
    --meta-model gpt-3.5-turbo \
    --meta-backend openai \
    --iterations 10
```

## Příklady pro různé úlohy

### GSM8K (matematické problémy)

```bash
uv run python main.py \
    --task gsm8k \
    --method opro \
    --model gpt-3.5-turbo \
    --meta-model gpt-3.5-turbo \
    --iterations 10 \
    --minibatch-size 20
```

### Claudette (ToS klasifikace - multi-label)

```bash
uv run python main.py \
    --task claudette \
    --method opro \
    --model gpt-3.5-turbo \
    --meta-model gpt-3.5-turbo \
    --iterations 10 \
    --minibatch-size 20
```

### Claudette Binary (ToS klasifikace - fair/unfair)

```bash
uv run python main.py \
    --task claudette_binary \
    --method opro \
    --model gpt-3.5-turbo \
    --meta-model gpt-3.5-turbo \
    --iterations 10 \
    --minibatch-size 20
```

## Výhody GPT-3.5-Turbo

1. **Rychlost** - API volání jsou velmi rychlá
2. **Cena** - Levnější než GPT-4
3. **Bez hardware požadavků** - Nepotřebujete GPU
4. **Kvalita** - Dobrá kvalita pro většinu úloh, zejména jako meta-optimizer

## Náklady

GPT-3.5-Turbo pricing (leden 2025):
- Input: ~$0.0005 / 1K tokenů
- Output: ~$0.0015 / 1K tokenů

Odhadované náklady pro typickou optimalizaci (10 iterací, 8 kandidátů, 20 examples):
- Přibližně 200-500 API volání
- Celkem ~$0.50-2.00 za běh

## Implementační detaily

### Architektura

```python
# src/llm_client.py obsahuje novou třídu:
class OpenAIClient(LLMClient):
    """LLM client using OpenAI's API"""

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a single prompt"""

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts"""
```

### Auto-detekce backendu

Factory funkce `create_llm_client()` automaticky detekuje backend:

```python
if "gpt" in model_name.lower():
    backend = "openai"
```

### Parametry

Podporované parametry pro OpenAIClient:
- `max_new_tokens` - maximální počet tokenů k vygenerování
- `temperature` - sampling temperature (0.0-2.0)
- Model automaticky používá chat completion API

## Troubleshooting

### API Key Error

```
ValueError: OPENAI_API_KEY not found in environment variables
```

**Řešení:** Zkontrolujte že `.env` soubor obsahuje váš API klíč.

### Import Error

```
ImportError: openai package not installed
```

**Řešení:** Spusťte `uv sync` pro instalaci dependencies.

### Rate Limiting

OpenAI má rate limity. Pokud dostanete chybu:
- Snižte počet kandidátů: `--num-candidates 4`
- Snižte počet iterací: `--iterations 5`
- Přidejte exponential backoff (implementováno v klientovi)

## Další kroky

- Experimentujte s hybridními nastaveními (lokální model + GPT API)
- Porovnejte výsledky GPT-3.5 vs Claude vs lokální modely
- Optimalizujte náklady použitím GPT-3.5 pro task a GPT-4 pouze pro meta-optimization

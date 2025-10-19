# GSM8K Evaluation Results - Srovnění Promptů

## Tabulka: Accuracy pro 1 Sample vs 5 Samples

| # | Prompt | 1 Sample | 5 Samples | Rozdíl | Zlepšení (%) |
|---|--------|----------|-----------|--------|---------------|
| 1 | Work through the problem piece by piece and box your final a | 88.32% | 93.71% | +5.38% | +6.1% |
| 2 | Work through the problem piece by piece. Box the final answe | 88.32% | 92.87% | +4.55% | +5.2% |
| 3 | Solve this math problem and present a structured solution. S | 87.87% | 91.96% | +4.09% | +4.7% |
| 4 | Please reason internally. On the last line, put your final a | 79.68% | 89.08% | +9.40% | +11.8% |
| 5 | (prázdný prompt) | 68.16% | 75.82% | +7.66% | +11.2% |
| 6 | Let's think step-by-step. | 64.44% | 68.69% | +4.25% | +6.6% |


## Detailní Tabulka s Úplnými Prompty

### 1. Prompt

**Prompt**: `Work through the problem piece by piece and box your final answer as #### NUMBER.`

| Metrika | 1 Sample | 5 Samples | Rozdíl |
|---------|----------|-----------|--------|
| Accuracy | 88.32% | 93.71% | **+5.38%** (+6.1%) |
| Correct | 1165 | 1236 | +71 |
| Total | 1319 | 1319 | — |

### 2. Prompt

**Prompt**: `Work through the problem piece by piece. Box the final answer as #### FINAL_ANSWER: NUMBER after completing all necessary calculations. Do not include units in your final answer; it should be just a number without any units. Once you have found the solution, conclude with 'Final Answer: #### NUMBER'.`

| Metrika | 1 Sample | 5 Samples | Rozdíl |
|---------|----------|-----------|--------|
| Accuracy | 88.32% | 92.87% | **+4.55%** (+5.2%) |
| Correct | 1165 | 1225 | +60 |
| Total | 1319 | 1319 | — |

### 3. Prompt

**Prompt**: `Solve this math problem and present a structured solution. State your final answer clearly at the end using the "#### NUMBER" format. Step 1: Perform calculation X; Step 2: Perform calculation Y; Final Answer: #### NUMBER`

| Metrika | 1 Sample | 5 Samples | Rozdíl |
|---------|----------|-----------|--------|
| Accuracy | 87.87% | 91.96% | **+4.09%** (+4.7%) |
| Correct | 1159 | 1213 | +54 |
| Total | 1319 | 1319 | — |

### 4. Prompt

**Prompt**: `Please reason internally. On the last line, put your final answer inside LaTeX: \boxed{<number>}. Output nothing else.`

| Metrika | 1 Sample | 5 Samples | Rozdíl |
|---------|----------|-----------|--------|
| Accuracy | 79.68% | 89.08% | **+9.40%** (+11.8%) |
| Correct | 1051 | 1175 | +124 |
| Total | 1319 | 1319 | — |

### 5. Prompt

**Prompt**: `(prázdný prompt)`

| Metrika | 1 Sample | 5 Samples | Rozdíl |
|---------|----------|-----------|--------|
| Accuracy | 68.16% | 75.82% | **+7.66%** (+11.2%) |
| Correct | 899 | 1000 | +101 |
| Total | 1319 | 1319 | — |

### 6. Prompt

**Prompt**: `Let's think step-by-step.`

| Metrika | 1 Sample | 5 Samples | Rozdíl |
|---------|----------|-----------|--------|
| Accuracy | 64.44% | 68.69% | **+4.25%** (+6.6%) |
| Correct | 850 | 906 | +56 |
| Total | 1319 | 1319 | — |


## Statistika

- **Celkem promptů**: 6
- **Prompty s oběma (1 a 5 samples)**: 6
- **Prompty pouze s 1 sample**: 0
- **Prompty pouze s 5 samples**: 0

### Zlepšení Self-Consistency (5 vs 1 sample)
- **Průměrné zlepšení**: +5.89%
- **Maximální zlepšení**: +9.40%
- **Minimální zlepšení**: +4.09%

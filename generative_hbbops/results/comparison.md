# HyLO Srovnání Experimentů

**Datum:** 2024-12-07
**Vzorků:** 20 (best)
**Baseline error rate:** 11.75% (inst=21, ex=9)
**Vec2Text:** ielabgroup/vec2text_gtr-base-st (opraveno)

## Výsledky

| # | Konfigurace | Strategie | Log-EI | Clip | Perturb | LeakyReLU | Final EI | Pred. Error | Cos Sim | Ex |
|---|-------------|-----------|--------|------|---------|-----------|----------|-------------|---------|-----|
| 1 | cd_baseline | CD | - | - | 0.1 | - | 0.00123 | 17.86% | 0.029 | 0 |
| 2 | cd_log_ei | CD | Yes | - | 0.1 | - | **0.00428** | 16.03% | 0.002 | 12 |
| 3 | cd_log_ei_clip | CD | Yes | 1.0 | 0.1 | - | **0.00503** | 17.36% | -0.078 | 9 |
| 4 | cd_log_ei_perturb | CD | Yes | - | 0.5 | - | 0.00428 | 16.03% | 0.002 | 12 |
| 5 | **cd_all** | CD | Yes | 1.0 | 1.0 | Yes | 0.00202 | **14.89%** | 0.021 | 7 |
| 6 | gs_baseline | GS | - | - | - | - | 0.00123 | 17.39% | 0.023 | 9 |
| 7 | gs_log_ei | GS | Yes | - | - | - | 0.00123 | 17.04% | 0.015 | 9 |
| 8 | gs_log_ei_clip | GS | Yes | 1.0 | - | - | 0.00123 | 17.39% | 0.063 | 9 |
| 9 | gs_leaky | GS | Yes | - | - | Yes | 0.00069 | 17.26% | 0.012 | 9 |
| 10 | **gs_all** | GS | Yes | 1.0 | - | Yes | 0.00069 | **16.99%** | 0.059 | 9 |

## Klíčová zjištění

### Nejlepší konfigurace
1. **cd_all** - 14.89% error rate (CD + všechna vylepšení)
2. gs_all - 16.99%
3. cd_log_ei - 16.03%

### Cosine Similarity (kvalita inverze)
- Všechny hodnoty jsou velmi nízké (< 0.1)
- Záporná hodnota u cd_log_ei_clip (-0.078) = embedding v opačném směru
- **Problém:** Optimalizované embeddingy jsou mimo distribuci Vec2Text

### Log-EI efekt
- CD: Zvyšuje EI 3-4× (0.00123 → 0.00428-0.00503)
- GS: Minimální efekt (EI zůstává 0.00123 nebo klesá)

### Poznámky
- Baseline (11.75%) stále nebyl překonán
- Všechny GS experimenty konvergují k exempláři 9
- CD má větší variabilitu ve výběru exempláře (0, 7, 9, 12)

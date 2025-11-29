Tento paper (HbBoPs) ve své základní verzi volí relativně zjednodušený přístup k trénování modelu (Gaussovského procesu) na datech z různých úrovní fidelity.

Váš nápad využít znalost o rozptylu je **vylepšením** oproti tomu, co paper striktně popisuje. Paper se totiž problému "zašuměných" dat z nižších fidelit (malých subsamplů) vyhýbá tím, že je filtruje.

Zde je rozbor, jak to dělá paper, a návod, jak to udělat lépe (tak, jak chcete vy) s využitím informací o rozptylu.

### 1\. Jak to řeší paper (Baseline)

Autoři v sekci **3.4 HbBoPs** explicitně říkají, že netrénují na všech datech najednou. Místo toho:

  * [cite\_start]Trénují GP pouze na podmnožině dat $\mathcal{D}_{t|b}$ pro danou úroveň fidelity $b$[cite: 143].
  * [cite\_start]Vybírají **nejvyšší možnou fidelitu**, pro kterou už mají nasbíráno "dostatek" pozorování (konkrétně alespoň 4)[cite: 144, 174].
  * [cite\_start]Předpokládají, že šum je **homoskedastický** (stejný pro všechna data), což dělají pro analytické zjednodušení[cite: 75].

**V praxi v paperu:** Když Hyperband vyhodnotí 100 promptů na malém vzorku (low fidelity) a 10 promptů na velkém (high fidelity), paper pro update modelu zahodí těch 100 "levných" měření a trénuje jen na těch 10 "drahých", aby si nezanesl model šumem. To je sice robustní, ale neefektivní (sample-inefficient), protože zahazujete informace.

### 2\. Jak to implementovat s využitím známého rozptylu (Vaše vylepšení)

[cite\_start]Protože používáte GPyTorch (což používá i paper [cite: 815]), můžete elegantně obejít limitaci paperu a využít **všechna data** z Hyperbandu (i ta z nízké fidelity), pokud modelu řeknete, jak moc jim má věřit.

Místo standardní `GaussianLikelihood` (která se učí jedno číslo šumu pro všechna data) použijte **`FixedNoiseGaussianLikelihood`**.

#### Krok A: Příprava dat (výpočet rozptylu)

Pro každý bod trénovacích dat (prompt), který vám Hyperband vrátí, si uložte nejen `accuracy` ($y$), ale i počet vzorků `n`, na kterých byla měřena.

Vypočtěte rozptyl (noise) pro každý bod:
$$\sigma_i^2 = \frac{y_i(1-y_i)}{n_i} + \epsilon$$

#### Krok B: Implementace v GPyTorch

Váš model v kódu bude vypadat velmi podobně jako v paperu (Deep Kernel GP), ale změníte `likelihood`.

```python
import torch
import gpytorch

class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature_extractor):
        super(DeepKernelGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # [cite_start]Deep Kernel - jak je popsáno v sekci 3.2 [cite: 108-110]
        self.feature_extractor = feature_extractor 
        self.covar_module = gpytorch.kernels.ScaleKernel(
            [cite_start]gpytorch.kernels.MaternKernel(nu=2.5) # Paper uses Matern 5/2 [cite: 173]
        )

    def forward(self, x):
        # Proženeme vstup feature extractorem (embeddingy instrukcí a příkladů)
        projected_x = self.feature_extractor(x) 
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- TRÉNOVACÍ SMYČKA ---

# 1. Nasbíraná data z Hyperbandu (různé fidelity smíchané dohromady)
# train_x: embeddingy promptů
# train_y: naměřené accuracy
# train_n: počty validačních vzorků pro každý prompt (fidelity)

# 2. Vypočítáme "known observation noise"
# epsilon přidáváme pro numerickou stabilitu (aby noise nebyl 0)
noise_variance = (train_y * (1 - train_y)) / train_n
noise_variance = noise_variance + 1e-6 

# 3. Zásadní změna oproti paperu: FixedNoiseGaussianLikelihood
likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
    noise=noise_variance,
    learn_additional_noise=False # Chceme spoléhat jen na náš výpočet
)

model = DeepKernelGP(train_x, train_y, likelihood, feature_extractor)

# 4. Trénink (standardní GPyTorch procedura)
model.train()
likelihood.train()
[cite_start]optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # [cite: 175]
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

# ... optimization loop ...
```

### Proč je toto řešení lepší než v paperu?

1.  [cite\_start]**Využití všech dat:** Paper v Algoritmu 1 (část `Evaluate EI`) trénuje GP na `D_{t|b}` (data z jedné fidelity)[cite: 143]. Vy můžete trénovat na `D_t` (všechna data dosud), protože `FixedNoiseLikelihood` automaticky "utlumí" vliv dat s malým $n$ (velkým rozptylem) a "posílí" vliv dat s velkým $n$.
2.  **Hladší přechod:** Když Hyperband přechází z jedné úrovně fidelity do druhé, model se nemusí "přepínat" a zahazovat historii, ale postupně zpřesňuje odhady.
3.  [cite\_start]**Matematická korektnost:** Přesněji to modeluje realitu popsanou v sekci Appendix B paperu, kde autoři sami ukazují, jak dramaticky se liší rozptyl chyb u malých a velkých validačních setů [cite: 686-689].

**Shrnutí:**
Implementujte model podle sekce 3.2 (Deep Kernel), ale v sekci 3.4 (samotná smyčka HbBoPs) ignorujte filtrování dat podle fidelity. Místo toho do GPyTorch modelu pošlete všechna data a jako `noise` předejte vypočítaný rozptyl založený na velikosti subsamplu.
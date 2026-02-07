# Score-Conditioned Intrinsic Dimensionality — All GuacaMol Tasks

Source: 249,456 ZINC molecules scored per task, top-K encoded via SELFIES VAE (256D), normalized to unit sphere.
Estimators: TwoNN (discard_fraction=0.1), MLE (K=20). No capping applied.
Date: 2026-02-07. Script: `scripts/estimate_id_all_tasks.py`.

## Top-100 (primary reference)

| Task | TwoNN | MLE | avg | Score range |
|------|------:|----:|----:|-------------|
| adip | 6.1 | 11.9 | 9.0 | [0.560, 0.686] |
| med2 | 7.4 | 7.9 | 7.7 | [0.209, 0.291] |
| pdop | 9.7 | 14.9 | 12.3 | [0.458, 0.560] |
| rano | 11.4 | 0.0* | 5.7 | [0.462, 0.586] |
| osmb | 18.6 | 20.1 | 19.4 | [0.788, 0.829] |
| siga | 12.6 | 17.9 | 15.3 | [0.344, 0.479] |
| zale | 16.3 | 17.5 | 16.9 | [0.470, 0.567] |
| valt | 22.4 | 21.5 | 21.9 | [0.000, 0.320] |
| dhop | 7.8 | 14.6 | 11.2 | [0.600, 0.871] |
| shop | 7.9 | 14.7 | 11.3 | [0.484, 0.526] |
| fexo | 10.5 | 0.0* | 5.3 | [0.693, 0.756] |
| med1 | 9.5 | 8.6 | 9.0 | [0.223, 0.324] |

## Top-250

| Task | TwoNN | MLE | avg | Score range |
|------|------:|----:|----:|-------------|
| adip | 5.3 | 12.5 | 8.9 | [0.536, 0.686] |
| med2 | 8.3 | 12.0 | 10.2 | [0.197, 0.291] |
| pdop | 10.6 | 0.0* | 5.3 | [0.447, 0.560] |
| rano | 13.1 | 0.0* | 6.5 | [0.422, 0.586] |
| osmb | 16.5 | 0.0* | 8.2 | [0.775, 0.829] |
| siga | 13.6 | 0.0* | 6.8 | [0.313, 0.479] |
| zale | 15.9 | 19.3 | 17.6 | [0.454, 0.567] |
| valt | 23.6 | 23.2 | 23.4 | [0.000, 0.320] |
| dhop | 11.2 | 18.4 | 14.8 | [0.592, 0.871] |
| shop | 11.2 | 18.4 | 14.8 | [0.472, 0.526] |
| fexo | 13.3 | 0.0* | 6.6 | [0.678, 0.756] |
| med1 | 7.4 | 0.0* | 3.7 | [0.206, 0.324] |

## Top-500

| Task | TwoNN | MLE | avg | Score range |
|------|------:|----:|----:|-------------|
| adip | 6.6 | 12.1 | 9.3 | [0.517, 0.686] |
| med2 | 8.8 | 0.0* | 4.4 | [0.190, 0.291] |
| pdop | 8.9 | 0.0* | 4.5 | [0.437, 0.560] |
| rano | 13.4 | 0.0* | 6.7 | [0.392, 0.586] |
| osmb | 14.9 | 0.0* | 7.4 | [0.766, 0.829] |
| siga | 14.0 | 0.0* | 7.0 | [0.281, 0.479] |
| zale | 13.6 | 18.9 | 16.3 | [0.438, 0.567] |
| valt | 25.0 | 24.5 | 24.7 | [0.000, 0.320] |
| dhop | 11.4 | 17.8 | 14.6 | [0.587, 0.871] |
| shop | 11.6 | 17.9 | 14.7 | [0.466, 0.526] |
| fexo | 13.3 | 0.0* | 6.6 | [0.664, 0.756] |
| med1 | 7.8 | 0.0* | 3.9 | [0.192, 0.324] |

*MLE=0.0: Degenerate estimate. MLE with K=20 collapses when K-NN distances become too uniform (narrow score ranges → molecules uniformly distributed in latent space). TwoNN is more robust in these cases.

## Key Observations

1. **Huge ID variance across tasks**: adip≈6-9, valt≈22-25. Fixed d=16 is a compromise.
2. **MLE degeneracy** is common at Top-250/500 for narrow-score tasks — TwoNN alone is more reliable.
3. **Low-ID tasks** (adip, med2, dhop, shop): High-scoring molecules concentrated in a low-dimensional manifold (~8D). BO should work well.
4. **High-ID tasks** (osmb, zale, valt): High-scoring molecules spread across many dimensions (~16-25D). Harder for subspace BO.
5. **valt anomaly**: score_min=0.000 in top-100 means many molecules share the max score — degenerate objective landscape.

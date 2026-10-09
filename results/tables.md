**T466** (427 metabolites evaluated on 65 NLIBD samples; permutation-null thresholds: D1 ρ > 0.360, D1+MGM ρ > 0.360, D1+MGM-arch ρ > 0.360, D2 ρ > 0.416)

| Model | Data | Mean ρ | 95% CI | Median ρ | Top-50 ρ | N(ρ > 0.5) | N(ρ > null) | NaN ρ | CV mean ρ |
|---|---|---|---|---|---|---|---|---|---|
| mean | D1 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.0 | 0.0 | 427.0 | 0.000 |
| ridge | D1 | 0.252 | [0.194, 0.319] | 0.242 | 0.589 | 50.0 | 126.0 | 0.0 | 0.411 |
| enet | D1 | 0.255 | [0.201, 0.317] | 0.240 | 0.587 | 48.0 | 125.0 | 4.0 | 0.391 |
| rf | D1 | 0.199 ± 0.005 | [0.149, 0.257] | 0.217 ± 0.005 | 0.551 ± 0.005 | 35.2 ± 3.3 | 114.2 ± 1.3 | 0.0 | 0.403 |
| mlp | D1 | 0.231 ± 0.008 | [0.174, 0.294] | 0.221 ± 0.012 | 0.580 ± 0.010 | 44.2 ± 3.6 | 117.4 ± 9.7 | 0.0 | 0.405 |
| mgm_frozen_ridge | D1+MGM | 0.290 | [0.233, 0.350] | 0.283 | 0.597 | 60.0 | 152.0 | 0.0 | 0.360 |
| mgm_random_frozen_ridge | D1+MGM-arch | 0.223 ± 0.015 | [0.168, 0.281] | 0.220 ± 0.019 | 0.515 ± 0.023 | 26.8 ± 8.0 | 92.8 ± 16.5 | 0.0 | 0.319 |
| mgm_frozen_mlp | D1+MGM | 0.201 ± 0.014 | [0.153, 0.254] | 0.192 ± 0.014 | 0.521 ± 0.012 | 25.8 ± 4.8 | 88.6 ± 11.3 | 0.0 | 0.344 |
| mgm_random_frozen_mlp | D1+MGM-arch | 0.213 ± 0.020 | [0.166, 0.264] | 0.210 ± 0.020 | 0.507 ± 0.037 | 24.2 ± 11.8 | 91.2 ± 22.0 | 0.0 | 0.304 |
| mgm_finetune | D1+MGM | 0.277 ± 0.004 | [0.222, 0.333] | 0.271 ± 0.007 | 0.573 ± 0.007 | 48.2 ± 4.0 | 141.2 ± 8.8 | 0.0 | 0.340 |
| mgm_scratch | D1+MGM-arch | 0.252 ± 0.008 | [0.191, 0.314] | 0.256 ± 0.003 | 0.533 ± 0.009 | 30.4 ± 4.0 | 118.6 ± 8.0 | 0.0 | 0.312 |
| d2_ridge | D2 | 0.357 | [0.308, 0.411] | 0.341 | 0.685 | 104.0 | 161.0 | 0.0 | 0.440 |
| e4c_scratch | D2 | 0.328 ± 0.008 | [0.277, 0.384] | 0.327 ± 0.004 | 0.670 ± 0.006 | 96.0 ± 4.8 | 147.2 ± 9.2 | 0.0 | 0.433 |
| e4c_selfsup | D2 | 0.308 ± 0.017 | [0.260, 0.362] | 0.302 ± 0.022 | 0.654 ± 0.005 | 87.4 ± 5.8 | 136.8 ± 14.0 | 0.0 | 0.416 |
| e4c_ihmp | D2 | 0.330 ± 0.015 | [0.281, 0.385] | 0.323 ± 0.020 | 0.661 ± 0.010 | 93.6 ± 3.8 | 141.6 ± 8.7 | 0.0 | 0.427 |
| e4c_selfsup_ihmp | D2 | 0.304 ± 0.015 | [0.255, 0.357] | 0.295 ± 0.017 | 0.642 ± 0.019 | 79.2 ± 11.1 | 128.4 ± 12.1 | 0.0 | 0.405 |

**TALL** (6948 metabolites evaluated on 65 NLIBD samples; permutation-null thresholds: D1 ρ > 0.371, D1+MGM ρ > 0.371, D1+MGM-arch ρ > 0.371)

| Model | Data | Mean ρ | 95% CI | Median ρ | Top-50 ρ | N(ρ > 0.5) | N(ρ > null) | NaN ρ | CV mean ρ |
|---|---|---|---|---|---|---|---|---|---|
| mean | D1 | 0.000 | [0.000, 0.000] | 0.000 | 0.000 | 0.0 | 0.0 | 6948.0 | 0.000 |
| ridge | D1 | 0.257 | [0.211, 0.312] | 0.263 | 0.752 | 785.0 | 2065.0 | 3.0 | 0.384 |
| mlp | D1 | 0.243 ± 0.010 | [0.197, 0.297] | 0.243 ± 0.012 | 0.753 ± 0.008 | 783.2 ± 57.4 | 1939.2 ± 110.3 | 3.0 | 0.384 |
| mgm_frozen_ridge | D1+MGM | 0.265 | [0.217, 0.317] | 0.266 | 0.726 | 806.0 | 2053.0 | 3.0 | 0.338 |
| mgm_random_frozen_ridge | D1+MGM-arch | 0.218 ± 0.012 | [0.171, 0.270] | 0.218 ± 0.012 | 0.662 ± 0.020 | 422.2 ± 95.0 | 1382.4 ± 181.8 | 3.0 | 0.303 |
| mgm_frozen_mlp | D1+MGM | 0.205 ± 0.011 | [0.160, 0.257] | 0.205 ± 0.011 | 0.658 ± 0.017 | 499.0 ± 70.3 | 1513.8 ± 137.0 | 3.0 | 0.335 |
| mgm_random_frozen_mlp | D1+MGM-arch | 0.211 ± 0.014 | [0.164, 0.263] | 0.210 ± 0.015 | 0.656 ± 0.030 | 432.8 ± 149.2 | 1440.0 ± 226.8 | 3.0 | 0.317 |
| mgm_finetune | D1+MGM | 0.250 ± 0.003 | [0.205, 0.299] | 0.252 ± 0.003 | 0.709 ± 0.004 | 708.6 ± 26.6 | 1918.4 ± 48.2 | 3.0 | 0.321 |
| mgm_scratch | D1+MGM-arch | 0.216 ± 0.013 | [0.161, 0.273] | 0.217 ± 0.013 | 0.645 ± 0.015 | 414.0 ± 90.4 | 1505.0 ± 147.2 | 3.0 | 0.336 |

**S176** (172 metabolites evaluated on 65 NLIBD samples; permutation-null thresholds: D2 ρ > 0.425)

| Model | Data | Mean ρ | 95% CI | Median ρ | Top-50 ρ | N(ρ > 0.5) | N(ρ > null) | NaN ρ | CV mean ρ |
|---|---|---|---|---|---|---|---|---|---|
| d2_ridge | D2 | 0.385 | [0.337, 0.439] | 0.400 | 0.605 | 48.0 | 72.0 | 0.0 | 0.461 |
| e4c_scratch | D2 | 0.378 ± 0.007 | [0.333, 0.429] | 0.378 ± 0.013 | 0.597 ± 0.007 | 46.2 ± 2.3 | 68.8 ± 3.9 | 0.0 | 0.445 |
| e4c_selfsup | D2 | 0.370 ± 0.011 | [0.328, 0.421] | 0.368 ± 0.011 | 0.588 ± 0.012 | 42.4 ± 3.8 | 65.0 ± 5.2 | 0.0 | 0.426 |
| e4c_ihmp | D2 | 0.385 ± 0.005 | [0.340, 0.437] | 0.387 ± 0.008 | 0.595 ± 0.007 | 46.2 ± 2.9 | 71.0 ± 3.7 | 0.0 | 0.454 |
| e4c_selfsup_ihmp | D2 | 0.369 ± 0.007 | [0.325, 0.420] | 0.377 ± 0.017 | 0.584 ± 0.012 | 40.8 ± 4.5 | 65.2 ± 7.0 | 0.0 | 0.454 |

**Paired comparisons on NLIBD** (paired bootstrap over test samples)

| Question | A | B | Targets | Δ mean ρ (A − B) | 95% CI | P(Δ ≤ 0) |
|---|---|---|---|---|---|---|
| H1 (E3): MGM's pretrained GPT-2 vs the same GPT-2 trained from random weights | mgm_finetune | mgm_scratch | T466 | +0.025 | [+0.000, +0.047] | 0.012 |
| H1 (E3): MGM's pretrained GPT-2 vs the same GPT-2 trained from random weights | mgm_finetune | mgm_scratch | TALL | +0.034 | [+0.010, +0.058] | 0.002 |
| H1 (E2): frozen MGM embeddings vs a random-weights GPT-2 on the same tokens (ridge head) | mgm_frozen_ridge | mgm_random_frozen_ridge | T466 | +0.067 | [+0.037, +0.094] | 0.000 |
| H1 (E2): frozen MGM embeddings vs a random-weights GPT-2 on the same tokens (ridge head) | mgm_frozen_ridge | mgm_random_frozen_ridge | TALL | +0.047 | [+0.027, +0.066] | 0.000 |
| H1 (E2): frozen MGM embeddings vs a random-weights GPT-2 on the same tokens (MLP head) | mgm_frozen_mlp | mgm_random_frozen_mlp | T466 | -0.012 | [-0.048, +0.022] | 0.756 |
| H1 (E2): frozen MGM embeddings vs a random-weights GPT-2 on the same tokens (MLP head) | mgm_frozen_mlp | mgm_random_frozen_mlp | TALL | -0.006 | [-0.033, +0.020] | 0.664 |
| H1 (E4): self-supervised pretraining on 1,503 source metagenomes vs scratch | e4c_selfsup | e4c_scratch | T466 | -0.020 | [-0.039, -0.001] | 0.979 |
| H1 (E4): self-supervised pretraining on 1,503 source metagenomes vs scratch | e4c_selfsup | e4c_scratch | S176 | -0.008 | [-0.022, +0.005] | 0.903 |
| H1 (E4): supervised pretraining on iHMP vs scratch | e4c_ihmp | e4c_scratch | T466 | +0.003 | [-0.008, +0.014] | 0.335 |
| H1 (E4): supervised pretraining on iHMP vs scratch | e4c_ihmp | e4c_scratch | S176 | +0.006 | [-0.005, +0.018] | 0.155 |
| H1 (E4): self-supervised then iHMP pretraining vs scratch | e4c_selfsup_ihmp | e4c_scratch | T466 | -0.024 | [-0.041, -0.006] | 0.998 |
| H1 (E4): self-supervised then iHMP pretraining vs scratch | e4c_selfsup_ihmp | e4c_scratch | S176 | -0.009 | [-0.026, +0.008] | 0.877 |
| Context: the E1 MLP vs ridge on the same species | mlp | ridge | T466 | -0.021 | [-0.038, -0.004] | 0.995 |
| Context: the E1 MLP vs ridge on the same species | mlp | ridge | TALL | -0.014 | [-0.028, +0.002] | 0.973 |
| Context: fine-tuned MGM vs ridge on the D1 species | mgm_finetune | ridge | T466 | +0.025 | [-0.015, +0.063] | 0.083 |
| Context: fine-tuned MGM vs ridge on the D1 species | mgm_finetune | ridge | TALL | -0.007 | [-0.035, +0.017] | 0.699 |

**Learning curve** (mean ρ on NLIBD, T466; mean ± SD over 5 seeds)

| Model | Data | n = 20 | n = 40 | n = 80 | n = 155 |
|---|---|---|---|---|---|
| ridge | D1 | 0.208 ± 0.015 | 0.210 ± 0.009 | 0.229 ± 0.011 | 0.252 ± 0.000 |
| mlp | D1 | 0.188 ± 0.025 | 0.204 ± 0.011 | 0.215 ± 0.015 | 0.231 ± 0.008 |
| mgm_frozen_ridge | D1+MGM | 0.180 ± 0.024 | 0.218 ± 0.021 | 0.257 ± 0.006 | 0.290 ± 0.000 |
| mgm_finetune | D1+MGM | 0.168 ± 0.020 | 0.208 ± 0.020 | 0.241 ± 0.008 | 0.277 ± 0.004 |
| mgm_scratch | D1+MGM-arch | 0.123 ± 0.049 | 0.151 ± 0.031 | 0.205 ± 0.025 | 0.252 ± 0.008 |
| e4c_scratch | D2 | 0.223 ± 0.019 | 0.247 ± 0.027 | 0.299 ± 0.011 | 0.328 ± 0.008 |
| e4c_ihmp | D2 | 0.221 ± 0.020 | 0.249 ± 0.024 | 0.298 ± 0.013 | 0.330 ± 0.015 |

**Ablations** (external = NLIBD)

| Ablation | Arm | Model | Scored on | Eval space | CV mean ρ | External mean ρ | N(ρ > 0.5) | Note |
|---|---|---|---|---|---|---|---|---|
| A1_input_transform | clr | ridge | T466 | log10 | 0.411 | 0.252 | 50.0 | main configuration (copied) |
| A1_input_transform | clr | mlp | T466 | log10 | 0.405 | 0.231 ± 0.008 | 44.2 | main configuration (copied) |
| A1_input_transform | log | ridge | T466 | log10 | 0.414 | 0.253 | 45.0 |  |
| A1_input_transform | log | mlp | T466 | log10 | 0.411 | 0.230 ± 0.009 | 42.8 |  |
| A2_target_transform | log10 | ridge | T466 | log10 | 0.411 | 0.252 | 50.0 | main configuration (copied) |
| A2_target_transform | log10 | mlp | T466 | log10 | 0.405 | 0.231 ± 0.008 | 44.2 | main configuration (copied) |
| A2_target_transform | clr | ridge | T466 | clr | 0.422 | 0.261 | 43.0 |  |
| A2_target_transform | clr | ridge | T466 | log10 | 0.347 | 0.218 | 34.0 |  |
| A2_target_transform | clr | mlp | T466 | clr | 0.415 | 0.241 ± 0.012 | 41.2 |  |
| A2_target_transform | clr | mlp | T466 | log10 | 0.336 | 0.201 ± 0.007 | 36.6 |  |
| A3_prevalence | 0.05 | ridge | T466 | log10 | 0.406 | 0.254 | 47.0 |  |
| A3_prevalence | 0.05 | mlp | T466 | log10 | 0.392 | 0.223 ± 0.015 | 38.2 |  |
| A3_prevalence | 0.10 | ridge | T466 | log10 | 0.411 | 0.252 | 50.0 | main configuration (copied) |
| A3_prevalence | 0.10 | mlp | T466 | log10 | 0.405 | 0.231 ± 0.008 | 44.2 | main configuration (copied) |
| A3_prevalence | 0.20 | ridge | T466 | log10 | 0.393 | 0.255 | 52.0 |  |
| A3_prevalence | 0.20 | mlp | T466 | log10 | 0.373 | 0.235 ± 0.008 | 56.8 |  |
| A4_published_protocol | clr_x_plus_1 | ridge | T466_from_all | clr | 0.413 | 0.262 | 57.0 |  |
| A4_published_protocol | clr_x_plus_1 | mlp | T466_from_all | clr | 0.396 | 0.253 ± 0.010 | 54.0 |  |

# Driver Evaluation Suite

This experiment compares four controllers across setup, tire, and grip changes.

- Repeats per condition: 2
- CSV: `experiments\results\driver_evaluation.csv`

| Driver | Completion | Avg completed lap | Best lap | Avg off-track | Avg score |
|---|---:|---:|---:|---:|---:|
| rule_based | 100.0% | 7.942s | 7.667s | 0.00% | 8.442 |
| raw_neural | 83.3% | 14.403s | 8.917s | 30.34% | 126.510 |
| vae_latent | 100.0% | 13.194s | 6.467s | 13.10% | 39.894 |
| optimized_vae_latent | 100.0% | 10.883s | 5.983s | 11.82% | 35.031 |

## Condition Winners

- balanced: optimized_vae_latent (7.200s, score 7.700, off-track 0.00%)
- high_downforce: optimized_vae_latent (7.217s, score 7.717, off-track 0.00%)
- low_drag: optimized_vae_latent (7.183s, score 7.683, off-track 0.00%)
- cold_tires: rule_based (7.683s, score 8.183, off-track 0.00%)
- worn_tires: rule_based (7.683s, score 8.183, off-track 0.00%)
- wet_track: rule_based (9.250s, score 9.750, off-track 0.00%)

Interpretation: a driver that is fast only in the balanced condition may be overfit. A stronger controller should keep completing laps as grip, tires, and aero setup change.
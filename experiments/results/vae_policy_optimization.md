# Optimized VAE-Latent Policy

The optimizer mutates the trained VAE-latent policy and keeps candidates that improve the lap score.

- Baseline VAE policy lap: 8.850 s
- Optimized VAE policy lap: 7.200 s
- Lap-time improvement: 1.650 s
- Optimized off-track: 0.00%
- Optimized score: 7.600
- Optimized model: `models\optimized_vae_latent_policy.npz`
- History: `experiments\results\vae_policy_optimization.csv`

Reference baselines:
- Rule-based driver: 7.683 s
- Raw neural imitation driver: 8.950 s

This is evolutionary policy optimization, not PPO. It is intentionally small and easy to inspect.
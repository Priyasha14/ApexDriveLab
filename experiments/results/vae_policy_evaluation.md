# VAE-Latent Policy Evaluation

The VAE-latent driver uses an 8D VAE encoder as a learned state representation before the policy predicts controls.

- VAE model: `models\driving_state_vae_8d.npz`
- Policy model: `models\vae_latent_policy.npz`
- Completed laps: 1
- Elapsed time: 8.85 s
- Off-track samples: 0.00%
- Max tire grip usage: 1.00

Comparison:

- Rule-based driver: 7.68 s, 0.00% off-track
- Raw neural imitation driver: 8.95 s, 0.00% off-track
- VAE-latent neural driver: 8.85 s, 0.00% off-track

Interpretation: increasing the VAE from 2D to 8D made the representation much better for control. The VAE-latent policy is slightly faster than the raw neural imitation baseline in this run, but still slower than the rule-based expert. The next improvement is policy optimization or reinforcement learning using lap reward.

# VAE-Latent Policy Training

This policy uses the VAE encoder as a learned feature extractor, then learns controls from the latent vector.

- Demonstration samples: 12527
- Latent size: 8
- Final training loss: 0.005778
- Final validation loss: 0.005401
- VAE model: `models\driving_state_vae_8d.npz`
- Policy model: `models\vae_latent_policy.npz`
# VAE Driving-State Model

The VAE learns a compressed latent representation of simulator driving states.

- Demonstration samples: 12526
- Final reconstruction loss: 0.048523
- Final KL loss: 20.689188
- Saved model: `models\driving_state_vae_8d.npz`

Use the encoder output as a compact state vector for later policy learning or anomaly detection.
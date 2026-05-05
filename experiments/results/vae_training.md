# VAE Driving-State Model

The VAE learns a compressed latent representation of simulator driving states.

- Demonstration samples: 12526
- Final reconstruction loss: 0.263220
- Final KL loss: 12.732705
- Saved model: `models\driving_state_vae.npz`

Use the encoder output as a compact state vector for later policy learning or anomaly detection.
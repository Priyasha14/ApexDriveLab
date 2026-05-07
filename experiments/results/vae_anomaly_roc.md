# VAE Anomaly ROC

This treats clean rule-driver states as normal and low-grip/noisy-control driving states as anomalies.

- Normal samples: 1200
- Anomaly samples: 1200
- ROC AUC: 0.9993
- Best threshold accuracy: 0.9946
- Best reconstruction-error threshold: 0.223084
- Plot: `experiments\results\vae_anomaly_roc.png`

ROC/accuracy are meaningful here because anomaly detection is a binary classification task. They are not the right metric for raw VAE reconstruction by itself.
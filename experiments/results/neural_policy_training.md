# Neural Policy Training

The neural driver is trained with imitation learning. It observes the simulator state and learns to copy the rule-based driver's control outputs.

- Demonstration samples: 10022
- Final training loss: 0.003017
- Final validation loss: 0.003069
- Saved model: `models\neural_policy.npz`

This is a baseline neural network, not reinforcement learning. The next step would be to fine-tune it with repeated lap rewards.
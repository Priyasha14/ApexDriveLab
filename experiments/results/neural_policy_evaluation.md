# Neural Policy Evaluation

The trained neural driver was evaluated in closed-loop control, meaning the neural network controlled the car directly instead of only predicting actions on saved examples.

- Model: `models\neural_policy.npz`
- Completed laps: 1
- Elapsed time: 8.95 s
- Off-track samples: 0.00%
- Max tire grip usage: 1.00

Baseline comparison from the same headless simulator:

- Rule-based driver lap time: 7.68 s
- Neural imitation driver lap time: 8.95 s

Interpretation: the neural policy can complete a clean lap, but it is slower than the rule-based controller it learned from. That is expected for a first imitation-learning model. The next improvement is to optimize or fine-tune the neural policy using lap-time reward, off-track penalties, and grip-usage penalties.

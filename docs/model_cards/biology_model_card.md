---
model_name: Exponential Decay (Population/Substance Decay)
equation: dN/dt = -λ · N
inputs:
  - t (time, s)
  - λ (decay rate, 1/s)
  - N0 (initial quantity, mol)
outputs:
  - N(t) (quantity at time t)
accuracy: < 2% MAE vs analytical exponential function
use_case: Drug elimination modeling, radioactive decay, cell apoptosis simulation, epidemic extinction modeling
training_config:
  optimizer: adam
  epochs: 5000
  learning_rate: 0.001
  loss_weights: adaptive
known_limitations: Pure exponential decay assumes no replenishment. Continuous models may not perfectly reflect discrete population die-offs at extremely low numbers (stochastic effects).
---

# 4. Biology Model
Leveraging PINNs with the exponential decay differential equation to model declining population or substance amounts. This captures everything from pharmacology clearance limits to nuclear half-life.

**Mathematical Form:**
`dN/dt = -λ · N`

**Use Cases:**
- Radioactive decay
- Epidemic modeling
- Cell biology models

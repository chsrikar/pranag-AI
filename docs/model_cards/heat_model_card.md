---
model_name: Heat Equation Model
equation: ∂u/∂t = α · ∂²u/∂x²
inputs:
  - x (position, m)
  - t (time, s)
outputs:
  - u (temperature, K)
accuracy: < 5% L2 error vs analytical solution
use_case: Modeling heat diffusion in solid materials, thermal management, cooling system simulation
training_config:
  optimizer: adam
  epochs: 5000
  learning_rate: 0.001
  loss_weights: adaptive
known_limitations: Assumes constant thermal diffusivity (α). Not suitable for convective or radiative dominant domains without modification.
---

# 1. Heat Equation Model (1D Transient)
This model simulates the heat transfer through a 1D material over time using the physics-informed neural network (PINN). It employs DeepXDE for solving the partial differential equation without requiring manual grid meshing or autograd handling.

**Mathematical Form:**
`∂u/∂t = α · ∂²u/∂x²`

**Use Cases:**
- Thermal management
- Cooling system simulation
- Material science

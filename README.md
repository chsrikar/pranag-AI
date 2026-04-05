# Physics-Informed Neural Network (PINN) System

A modular, scalable PINN training system built using DeepXDE. It integrates Parquet-based data ingestion via DuckDB and lightweight surrogate modeling via PyTorch to simulate and solve multiscale multi-physics equations.

## Project Architecture
- `pinn_system/`
  - `data_loader.py` - DuckDB optimized data loader
  - `base_pinn_framework.py` - Reusable DeepXDE Model
  - `models/` - Standalone physics definitions
  - `adaptive/` - Adaptive Loss system
  - `surrogate/` - PyTorch inference surrogate
  - `train.py` - End-to-end framework execution
- `data/` - Dataset processing features
- `docs/model_cards/` - Markdown descriptions of underlying PDE models

## Setup

First install dependencies:
```bash
pip install -r requirements.txt
```

Generate dummy data to test out pipelines:
```bash
python data/generate_dummy_data.py
```

## Running the Training Pipeline

The whole framework connects through the CLI tool: `pinn_system/train.py`.

```bash
python pinn_system/train.py --model heat --epochs 5000 --lr 0.001 --data data/dataset.parquet
```

- `--model`: Pick your physics domain (`heat`, `stress`, `growth`, `biology`, `chemistry`).
- `--epochs`: Adjust training steps.
- `--lr`: Set the step-size learning rate optimizer parameter.
- `--data`: Specify input dataset paths.

### Available Models
1. **Heat**: 1D Transient Heat Equation (Temperature Simulation).
2. **Stress**: Structural Integrity under linear Hooke's rules.
3. **Growth**: Bounded ecological capacity dynamics.
4. **Biology**: Decay mechanisms and population turnover.
5. **Chemistry**: Chemical rates through Arrhenius principles.

View the `docs/model_cards` directory for details on parameters, accuracy, and use cases.

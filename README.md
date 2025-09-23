# MountainCar-v0
This project implements a Genetic Algorithm (GA) to solve the classic MountainCar-v0 environment from Gymnasium (Farama AI)
.

The MountainCar problem consists of driving a car out of a valley by building enough momentum to reach the goal at the top of a hill. Since the engine power is insufficient to go straight up, the agent must learn to move back and forth strategically to succeed.

## 🔬 Project Overview

- Chromosome Representation: encodes the parameters of a simple decision policy (e.g., weights for a linear model or a small neural network).

- Fitness Function: total reward accumulated across multiple simulation episodes.

### Genetic Algorithm Components:

- Selection, crossover, and mutation operators.

- Population of at least 30 individuals.

- Minimum of 50 generations.

- Experimentation: runs with at least 3 different parameter configurations (population size, mutation rate, crossover type, etc.).

- Evaluation: plots of average and maximum fitness per generation.

## 📊 Deliverables

- Source code with modular structure and documentation.

- Experimental results and visualizations.

- Technical report analyzing the performance of different GA configurations.

## 📂 Project Structure

| File / Folder            | Description                                                                                                      |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `pyproject.toml`         | Poetry configuration (dependencies and project metadata).                                                        |
| `README.md`              | General description of the project (GA applied to MountainCar-v0).                                               |
| **`src/`**               | Main source code directory.                                                                                      |
| ├── `main.py`            | Runs a single GA configuration, logs results and generates `training_curve.png`.                                 |
| ├── `ga.py`              | Genetic Algorithm implementation: population, selection, crossover, mutation, elitism.                           |
| ├── `policy.py`          | Policy definition: converts a chromosome into a simple NN (2→8→3 with ReLU) to decide actions.                   |
| ├── `utils.py`           | Utility functions (e.g., plotting best/avg fitness curves, saving PNG).                                          |
| **`src/experiments/`**   | Experiment-related scripts, logs, and outputs.                                                                   |
| ├── `results.py`         | Runs multiple GA configurations (baseline, high\_mutation, large\_population). Saves plots + CSVs in `/outputs`. |
| ├── `compare_results.py` | Reads all CSVs in `/outputs` and generates a single comparison graph (`comparison.png`).                         |
| ├── `logs.txt`           | Log file with fitness results per generation when running `main.py`.                                             |
| └── `outputs/`           | Auto-generated folder containing results: training curves, per-config plots, CSVs, and comparisons.              |

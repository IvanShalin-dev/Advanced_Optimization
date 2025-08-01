# 🧬 Optimization Benchmark: GA vs DE vs PSO

This project compares three population-based metaheuristic algorithms — **Genetic Algorithm (GA)**, **Differential Evolution (DE)**, and **Particle Swarm Optimization (PSO)** — for solving the 2D Rastrigin function. The objective is to evaluate their convergence behavior and optimization performance under identical conditions.

---

## 🚀 Algorithms Implemented

### ✅ Genetic Algorithm (GA)
- **Selection:** Tournament (k = 3)
- **Crossover:** Arithmetic crossover
- **Mutation:** Gaussian perturbation per gene
- **Elitism:** Enabled (best individual preserved)

### ✅ Differential Evolution (DE)
- **Mutation:** `mutant = a + F * (b - c)`
- **Crossover:** Binomial crossover with random index
- **Selection:** Greedy selection (trial vs. target)

### ✅ Particle Swarm Optimization (PSO)
- **Velocity Update:** Inertia + Cognitive + Social components
- **Adaptive Inertia:** Linearly decreasing from 0.9 → 0.4
- **Acceleration Coefficients:** `c1 = c2 = 1.5`

---

## ⚙️ Experimental Setup

| Parameter              | Value            |
|------------------------|------------------|
| Objective Function     | Rastrigin (2D)   |
| Search Bounds          | [-5.12, 5.12]    |
| Generations            | 200              |
| Population Size        | 50               |
| GA Mutation Rate       | 0.1              |
| GA Crossover Rate      | 0.8              |
| DE F (scale factor)    | 0.5              |
| DE CR (crossover rate) | 0.9              |
| PSO Inertia (w)        | 0.9 → 0.4        |
| PSO c1, c2             | 1.5              |
| Runs                   | Single run per algorithm |

---

## ▶️ How to Run

1. **Install dependencies** (if any):
   ```bash
   pip install -r requirements.txt
2. **Run all optimizers:**
     ```bash
   python main.py

3.	**Check the output:**
	•	Each algorithm logs its generation-wise best fitness to:
	•	data/ga_fitness_log.csv
	•	data/de_fitness_log.csv
	•	data/pso_fitness_log.csv

4.  **Run Plot Convergence:**
      ```bash
     python plot_convergence.py     

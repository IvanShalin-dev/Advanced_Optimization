import pandas as pd
import matplotlib.pyplot as plt

# Load convergence data
ga = pd.read_csv("data/ga_fitness_log.csv")
de = pd.read_csv("data/de_fitness_log.csv")
pso = pd.read_csv("data/pso_fitness_log.csv")

plt.figure(figsize=(10, 6))
plt.plot(ga["generation"], ga["fitness"], label="Genetic Algorithm", linewidth=2)
plt.plot(de["generation"], de["fitness"], label="Differential Evolution", linewidth=2)
plt.plot(pso["generation"], pso["fitness"], label="Particle Swarm Optimization", linewidth=2)

plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Convergence of GA, DE, and PSO on Rastrigin Function")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/convergence_plot.png")
plt.show()
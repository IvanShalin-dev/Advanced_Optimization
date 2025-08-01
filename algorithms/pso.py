import numpy as np
import pandas as pd
import os
from utils.rastrigin import rastrigin

class ParticleSwarmOptimization:
    def __init__(self, pop_size=50, generations=200, w_start=0.9, w_end=0.4,
                 c1=1.5, c2=1.5, bounds=[-5.12, 5.12]):
        """
        Initialize Particle Swarm Optimization parameters.
        """
        self.pop_size = pop_size
        self.generations = generations
        self.w_start = w_start  # Initial inertia weight
        self.w_end = w_end      # Final inertia weight
        self.c1 = c1            # Cognitive coefficient
        self.c2 = c2            # Social coefficient
        self.bounds = bounds
        self.dimension = 2
        self.history = []       # Store best fitness per generation

    def run(self, log_dir="data"):
        """
        Run the PSO algorithm and save generation-wise fitness log to CSV.

        Returns:
            gbest: coordinates of best solution
            self.history: list of best fitness values
        """
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "pso_fitness_log.csv")

        # Step 1: Initialization
        position = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, self.dimension))
        velocity = np.random.uniform(-1, 1, (self.pop_size, self.dimension))
        pbest = position.copy()
        pbest_fitness = np.array([rastrigin(ind) for ind in pbest])

        gbest_idx = np.argmin(pbest_fitness)
        gbest = pbest[gbest_idx].copy()

        # Step 2: Iterative optimization
        for gen in range(self.generations):
            # Linearly decrease inertia weight
            w = self.w_start - (self.w_start - self.w_end) * (gen / self.generations)

            for i in range(self.pop_size):
                r1, r2 = np.random.rand(self.dimension), np.random.rand(self.dimension)

                # Velocity update (inertia + cognitive + social)
                velocity[i] = (
                    w * velocity[i]
                    + self.c1 * r1 * (pbest[i] - position[i])
                    + self.c2 * r2 * (gbest - position[i])
                )

                # Position update
                position[i] += velocity[i]
                position[i] = np.clip(position[i], self.bounds[0], self.bounds[1])

                # Fitness evaluation
                fitness = rastrigin(position[i])
                if fitness < pbest_fitness[i]:
                    pbest[i] = position[i].copy()
                    pbest_fitness[i] = fitness

                    # Global best update
                    if fitness < rastrigin(gbest):
                        gbest = position[i].copy()

            # Record best fitness for this generation
            best_fit = rastrigin(gbest)
            self.history.append(best_fit)

            # Console output
            print(f"Generation {gen+1} - Best Fitness: {best_fit:.4f}")

        # Step 3: Save to CSV
        pd.DataFrame({
            "generation": list(range(1, self.generations + 1)),
            "fitness": self.history
        }).to_csv(csv_path, index=False)

        return gbest, self.history
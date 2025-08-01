import numpy as np
import pandas as pd
import os
from utils.rastrigin import rastrigin

class DifferentialEvolution:
    def __init__(self, pop_size=50, generations=200, F=0.5, CR=0.9,
                 bounds=[-5.12, 5.12]):
        """
        Initialize Differential Evolution parameters.
        """
        self.pop_size = pop_size
        self.generations = generations
        self.F = F  # Differential weight (mutation factor)
        self.CR = CR  # Crossover probability
        self.bounds = bounds
        self.dimension = 2  # Problem dimension (2D Rastrigin)
        self.history = []   # Stores best fitness of each generation

    def initialize_population(self):
        """
        Create initial population within search bounds.
        """
        return np.random.uniform(self.bounds[0], self.bounds[1],
                                 (self.pop_size, self.dimension))

    def evaluate_fitness(self, population):
        """
        Compute fitness for the entire population.
        """
        return np.array([rastrigin(ind) for ind in population])

    def mutate(self, population, idx):
        """
        Create a mutant vector using DE/rand/1 strategy.
        """
        indices = [i for i in range(self.pop_size) if i != idx]
        a, b, c = population[np.random.choice(indices, 3, replace=False)]
        mutant = a + self.F * (b - c)
        return np.clip(mutant, self.bounds[0], self.bounds[1])

    def crossover(self, target, mutant):
        """
        Perform binomial crossover between target and mutant vectors.
        """
        trial = np.copy(target)
        for i in range(self.dimension):
            if np.random.rand() < self.CR or i == np.random.randint(self.dimension):
                trial[i] = mutant[i]
        return trial

    def run(self, log_dir="data"):
        """
        Run the DE optimizer and log best fitness per generation to CSV.

        Returns:
            best_solution: coordinates of best individual
            self.history: list of best fitness values
        """
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "de_fitness_log.csv")

        # Step 1: Initialization
        population = self.initialize_population()
        fitness = self.evaluate_fitness(population)

        # Step 2: Evolution over generations
        for gen in range(self.generations):
            new_population = []

            for i in range(self.pop_size):
                target = population[i]
                mutant = self.mutate(population, i)
                trial = self.crossover(target, mutant)
                trial_fit = rastrigin(trial)

                # Greedy selection
                if trial_fit < fitness[i]:
                    new_population.append(trial)
                else:
                    new_population.append(target)

            # Update population and fitness
            population = np.array(new_population)
            fitness = self.evaluate_fitness(population)
            best_fit = np.min(fitness)
            self.history.append(best_fit)

            # Console output
            print(f"Generation {gen+1} - Best Fitness: {best_fit:.4f}")

        # Step 3: Save fitness history to CSV
        pd.DataFrame({
            "generation": list(range(1, self.generations + 1)),
            "fitness": self.history
        }).to_csv(csv_path, index=False)

        # Step 4: Return best individual
        best_idx = np.argmin(fitness)
        return population[best_idx], self.history
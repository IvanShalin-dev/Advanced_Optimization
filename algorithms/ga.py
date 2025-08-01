import numpy as np
import pandas as pd
import os
from utils.rastrigin import rastrigin

class GeneticAlgorithm:
    def __init__(self, pop_size=50, generations=200, mutation_rate=0.1,
                 crossover_rate=0.8, bounds=[-5.12, 5.12], elitism=True):
        """
        Initialize the Genetic Algorithm parameters.
        """
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.bounds = bounds
        self.elitism = elitism
        self.dimension = 2  # Problem is 2D (Rastrigin function)
        self.history = []   # Store best fitness per generation

    def initialize_population(self):
        """
        Generate initial population within the given bounds.
        """
        return np.random.uniform(self.bounds[0], self.bounds[1],
                                 (self.pop_size, self.dimension))

    def evaluate_fitness(self, population):
        """
        Compute fitness for each individual in the population.
        """
        return np.array([rastrigin(ind) for ind in population])

    def tournament_selection(self, population, fitness, k=3):
        """
        Perform tournament selection of size k.
        """
        selected = []
        for _ in range(self.pop_size):
            idx = np.random.choice(len(population), k)
            best = population[idx[np.argmin(fitness[idx])]]
            selected.append(best)
        return np.array(selected)

    def crossover(self, parent1, parent2):
        """
        Perform arithmetic crossover with a random weight.
        """
        if np.random.rand() < self.crossover_rate:
            alpha = np.random.rand()
            return alpha * parent1 + (1 - alpha) * parent2
        return parent1.copy()

    def mutate(self, individual):
        """
        Apply Gaussian mutation to each gene.
        """
        for i in range(self.dimension):
            if np.random.rand() < self.mutation_rate:
                individual[i] += np.random.normal(0, 0.1)
                individual[i] = np.clip(individual[i], self.bounds[0], self.bounds[1])
        return individual

    def run(self, log_dir="data"):
        """
        Run the Genetic Algorithm and log generation-wise fitness to CSV.
        
        Returns:
            best_solution: numpy array of best individual's coordinates
            self.history: list of best fitness values
        """
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "ga_fitness_log.csv")

        # Step 1: Initialize
        population = self.initialize_population()
        fitness = self.evaluate_fitness(population)

        # Step 2: Run evolution for N generations
        for gen in range(self.generations):
            new_population = []

            # Elitism: preserve best individual
            if self.elitism:
                elite_idx = np.argmin(fitness)
                elite = population[elite_idx].copy()

            # Selection via tournament
            selected = self.tournament_selection(population, fitness)

            # Crossover + Mutation
            for i in range(0, self.pop_size, 2):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % self.pop_size]
                child1 = self.mutate(self.crossover(parent1, parent2))
                child2 = self.mutate(self.crossover(parent2, parent1))
                new_population.append(child1)
                new_population.append(child2)

            # Trim to population size
            population = np.array(new_population[:self.pop_size])

            # Replace worst individual with elite
            if self.elitism:
                worst_idx = np.argmax(self.evaluate_fitness(population))
                population[worst_idx] = elite

            # Evaluate fitness
            fitness = self.evaluate_fitness(population)
            best_fit = np.min(fitness)
            self.history.append(best_fit)

            # Print to console
            print(f"Generation {gen+1} - Best Fitness: {best_fit:.4f}")

        # Step 3: Save fitness log as CSV
        pd.DataFrame({
            "generation": list(range(1, self.generations + 1)),
            "fitness": self.history
        }).to_csv(csv_path, index=False)

        # Step 4: Return best solution
        best_idx = np.argmin(fitness)
        best_solution = population[best_idx]
        return best_solution, self.history
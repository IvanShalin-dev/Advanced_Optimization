from algorithms.ga import GeneticAlgorithm
from algorithms.de import DifferentialEvolution
from algorithms.pso import ParticleSwarmOptimization

def run_and_log(optimizer, name, save_dir="data"):
    """
    Run a given optimizer and print final solution and fitness.
    CSV logging is already handled inside each optimizer's run() method.
    """
    best_solution, history = optimizer.run(log_dir=save_dir)
    print(f"\nBest solution found by {name}: {best_solution}")
    print(f"Final fitness: {history[-1]:.4f}\n")

def main():
    print("Running Genetic Algorithm...")
    run_and_log(GeneticAlgorithm(), "GA")

    print("Running Differential Evolution...")
    run_and_log(DifferentialEvolution(), "DE")

    print("Running Particle Swarm Optimization...")
    run_and_log(ParticleSwarmOptimization(), "PSO")

if __name__ == "__main__":
    main()
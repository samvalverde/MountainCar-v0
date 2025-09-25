import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from ga import GeneticAlgorithm
from policy import Policy

# poetry run python src/results.py

def run_experiment(config, env_name="MountainCar-v0", n_generations=50, episodes_per_individual=3, hidden_size=8):
    env = gym.make(env_name)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    chromosome_length = obs_size * hidden_size + hidden_size * action_size

    ga = GeneticAlgorithm(
        population_size=config["population_size"],
        chromosome_length=chromosome_length,
        mutation_rate=config["mutation_rate"],
        crossover_rate=config["crossover_rate"],
        elitism_rate=0.05,
        mating_pool_rate=0.25,
    )

    best_fitness_per_gen = []
    avg_fitness_per_gen = []

    for gen in range(n_generations):
        fitness_scores = []

        for chromosome in ga.population:
            policy = Policy(chromosome, obs_size, action_size, hidden_size)
            total_reward = 0

            for _ in range(episodes_per_individual):
                obs, _ = env.reset()
                done = False
                while not done:
                    action = policy.act(obs)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    total_reward += reward

            fitness_scores.append(total_reward / episodes_per_individual)

        best_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)

        best_fitness_per_gen.append(best_fitness)
        avg_fitness_per_gen.append(avg_fitness)

        print(f"[Config {config['name']}] Gen {gen+1}: Best={best_fitness}, Avg={avg_fitness:.2f}")

        ga.evolve(fitness_scores)

    env.close()
    return best_fitness_per_gen, avg_fitness_per_gen


def save_results(best, avg, config_name, out_dir="src/experiments/outputs"):
    os.makedirs(out_dir, exist_ok=True)

    # Guardar gráfica
    plt.figure()
    plt.plot(best, label="Best fitness")
    plt.plot(avg, label="Average fitness")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (reward)")
    plt.title(f"GA on MountainCar - {config_name}")
    plt.legend()
    path_png = os.path.join(out_dir, f"fitness_{config_name}.png")
    plt.savefig(path_png)
    plt.close()

    # Guardar CSV
    path_csv = os.path.join(out_dir, f"fitness_{config_name}.csv")
    with open(path_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Generation", "Best", "Average"])
        for i, (b, a) in enumerate(zip(best, avg), start=1):
            writer.writerow([i, b, a])

    print(f"Results saved: {path_png}, {path_csv}")


if __name__ == "__main__":
    configs = [
        {"name": "baseline_hidden", "population_size": 30, "mutation_rate": 0.1, "crossover_rate": 0.7},
        {"name": "high_mutation_hidden", "population_size": 30, "mutation_rate": 0.3, "crossover_rate": 0.7},
        {"name": "small_population_hidden", "population_size": 15, "mutation_rate": 0.1, "crossover_rate": 0.7},
        {"name": "high_crossover_rate_hidden", "population_size": 30, "mutation_rate": 0.1, "crossover_rate": 0.9}
    ]

    for config in configs:
        best, avg = run_experiment(config)
        save_results(best, avg, config["name"])

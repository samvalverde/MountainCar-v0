import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from ga import GeneticAlgorithm
from policy import Policy
from utils import plot_fitness
import sys

# poetry run python src/main.py

def evaluate_individual(env, policy, episodes=3):
    """Evalúa un cromosoma en múltiples episodios y devuelve el fitness promedio."""
    total_reward = 0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
    return total_reward / episodes


def main():

    # --- Configuración ---
    env_name = "MountainCar-v0"
    n_generations = 50
    population_size = 30
    episodes_per_individual = 3

    # Entrenamiento (sin renderizado)
    env = gym.make(env_name, render_mode=None, goal_velocity=0.0)

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    hidden_size = 8
    chromosome_length = obs_size * hidden_size + hidden_size * action_size

    ga = GeneticAlgorithm(
        population_size=population_size,
        chromosome_length=chromosome_length,
        mutation_rate=0.1,
        crossover_rate=0.7,
        elitism_rate=0.05,
        mating_pool_rate=0.25
    )

    best_fitness_per_gen = []
    avg_fitness_per_gen = []
    best_individual = None
    best_score = -np.inf

    log_file = open("src/experiments/logs.txt", "w")     # Registrar salida en logs
    sys.stdout = log_file

    for gen in range(n_generations):
        fitness_scores = []

        for chromosome in ga.population:
            policy = Policy(chromosome, obs_size, action_size, hidden_size)
            score = evaluate_individual(env, policy, episodes=episodes_per_individual)
            fitness_scores.append(score)

            # Guardar mejor individuo de todo el entrenamiento
            if score > best_score:
                best_score = score
                best_individual = chromosome.copy()

        best_fitness = np.max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)

        best_fitness_per_gen.append(best_fitness)
        avg_fitness_per_gen.append(avg_fitness)

        print(f"Gen {gen+1}: Best = {best_fitness}, Avg = {avg_fitness:.2f}")

        ga.evolve(fitness_scores)

    env.close()
    plot_fitness(best_fitness_per_gen, avg_fitness_per_gen)

    sys.stdout = sys.__stdout__
    log_file.close()
    print("Logs guardados!")

    # --- Test visual con el mejor individuo ---
    print("\nEvaluando el mejor individuo con renderizado gráfico...\n")

    env_visual = gym.make(env_name, render_mode="human", goal_velocity=0.0)    # Crear entorno con renderizado (human)

    best_policy = Policy(best_individual, obs_size, action_size)
    evaluate_individual(env_visual, best_policy, episodes=1)
    env_visual.close()


if __name__ == "__main__":
    main()
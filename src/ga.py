# src/ga.py
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, chromosome_length, mutation_rate, crossover_rate,
                 elitism_rate=0.05, mating_pool_rate=0.25):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.mating_pool_rate = mating_pool_rate
        self.population = np.random.uniform(-1, 1, (population_size, chromosome_length))

    def select(self, fitness, pool):
        i, j = np.random.randint(0, len(pool), 2)
        return pool[i] if fitness[i] > fitness[j] else pool[j]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.chromosome_length)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def mutate(self, chromosome):
        for i in range(self.chromosome_length):
            if np.random.rand() < self.mutation_rate:
                chromosome[i] += np.random.normal(0, 0.1)
        return chromosome

    def evolve(self, fitness):
        n_elite = max(1, int(self.population_size * self.elitism_rate))
        elite_indices = np.argsort(fitness)[-n_elite:]
        elites = self.population[elite_indices].copy()

        n_pool = max(2, int(self.population_size * self.mating_pool_rate))
        pool_indices = np.argsort(fitness)[-n_pool:]
        mating_pool = self.population[pool_indices]

        new_population = list(elites)

        while len(new_population) < self.population_size:
            parent1 = self.select(fitness, mating_pool)
            parent2 = self.select(fitness, mating_pool)
            child1, child2 = self.crossover(parent1, parent2)
            new_population.append(self.mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self.mutate(child2))

        self.population = np.array(new_population)

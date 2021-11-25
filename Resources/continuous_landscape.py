from __future__ import annotations

import copy
from itertools import product
import threading

import numpy as np
from matplotlib import pyplot as plt


def main():

    fitnesses = list()
    for _ in range(4):
        landscape = Landscape(20, 0, 20)
        fitness_over_time, best_soln, gen_best_soln = \
            find_solution(landscape, generations=100)
        fitnesses.append(fitness_over_time)
    fitness_over_time = np.mean(fitnesses, axis=0)
    plt.plot(list(range(len(fitness_over_time))), fitness_over_time,
             label='k=0')

    fitnesses = list()
    for _ in range(4):
        landscape = Landscape(20, 10, 20)
        fitness_over_time, best_soln, gen_best_soln = \
            find_solution(landscape, generations=100)
        fitnesses.append(fitness_over_time)
    fitness_over_time = np.mean(fitnesses, axis=0)
    plt.plot(list(range(len(fitness_over_time))), fitness_over_time,
             label='k=10')

    fitnesses = list()
    for _ in range(4):
        landscape = Landscape(20, 19, 20)
        fitness_over_time, best_soln, gen_best_soln = \
            find_solution(landscape, generations=100)
        fitnesses.append(fitness_over_time)
    fitness_over_time = np.mean(fitnesses, axis=0)
    plt.plot(list(range(len(fitness_over_time))), fitness_over_time,
             label='k=19')

    plt.legend()
    plt.show()

    # for i in range(20):
    #     genome = list(np.random.randint(0, 10, 20))
    #     print(landscape.get_fitness(genome))
    #     _, fitness, __ = landscape.find_nearest_optima(genome)
    #     print(fitness,'\n')

    # test_vals = list(range(20))
    # combinations = product(test_vals, repeat=20)
    # best_genomes = list()
    # best_fitnesses = list()
    # for i, combination in enumerate(combinations):
    #     genome = list(combination)
    #     optimal_genome, fitness, __ = landscape.find_nearest_optima(genome)
    #     if optimal_genome not in best_genomes:
    #         best_genomes.append(optimal_genome)
    #         best_fitnesses.append(fitness)
    #     if i % 100 == 0:
    #         print(f"{i} of {10**20} combinations tested.")


class Landscape:

    def __init__(self,
                 n: int = 20,
                 k: int = 10,
                 max_gene_val: int = 9) -> None:
        """
        Creates a lookup table of floats in the range [0, 1) that describe
        the correlation of two genes, with smaller values describing less
        correlation and higher values describing higher correlation.
        :param n: The length of the genome.
        :param k: The number of other locations each gene interacts with.
        This is the same for all genes.
        :param max_gene_val: The number of integer values a gene can take, ranging from
        0 to r-1.
        """
        if k >= n or k < 0:
            raise ValueError('k must be an integer greater than or equal to 0 '
                             'ans less than genome length n.')
        self.n = n
        self.k = k
        self.max_gene_val = max_gene_val
        self.gene_contribution_weight_matrix = np.random.rand(n, n)

    def get_fitness(self, genome: list[int]) -> float:
        """
        Takes a genome and returns the fitness of the genome. Fitness is the
        average of the contribution of each gene.
        :param genome: The genome to evaluate the fitness of.
        :return:
        """
        gene_values = list()
        for loc in range(len(genome)):
            contributing_gene_values = list()
            for i in range(self.k + 1):
                contributing_gene_values.append(genome[(loc + i) % self.n])
            weights = list()
            for i in range(self.k + 1):
                weights.append(
                    self.gene_contribution_weight_matrix
                    [loc][(loc + i) % self.n]
                )
            gene_values.append(np.multiply(contributing_gene_values, weights))
            pass
        return float(np.mean(gene_values))

    def find_nearest_optima(self,
                            solution):
        """
        Basic hill climber.
        """

        # solution = np.array(np.random.randint(2, size=landscape.n))
        solution_fitness = self.get_fitness(solution)
        steps_to_solution = 0
        local_optima_found = False

        while not local_optima_found:
            candidate = copy.deepcopy(solution)
            candidate_fitness = self.get_fitness(candidate)
            local_optima_found = True

            # Check all locations that require incrementing a gene value by 1
            for i in range(self.n):
                if solution[i] == self.max_gene_val:
                    continue
                step = copy.deepcopy(solution)
                step[i] += 1
                step_fitness = self.get_fitness(step)
                if step_fitness > candidate_fitness:
                    candidate = step
                    candidate_fitness = step_fitness
                    local_optima_found = False

            # Check all locations that require decrementing a gene value by 1
            for i in range(self.n):
                if solution[i] == 0:
                    continue
                step = copy.deepcopy(solution)
                step[i] -= 1
                step_fitness = self.get_fitness(step)
                if step_fitness > candidate_fitness:
                    candidate = step
                    candidate_fitness = step_fitness
                    local_optima_found = False

            if local_optima_found is False:
                solution = candidate
                solution_fitness = candidate_fitness
                steps_to_solution += 1

        return solution, solution_fitness, steps_to_solution


class Individual:

    def __init__(self,
                 fitness_fx: callable,
                 n: int,
                 max_gene_val: int):
        self.fitness_fx = fitness_fx
        self.max_gene_val = max_gene_val
        self.genome = list(np.array(np.random.randint(0, max_gene_val, n)))

    def fitness(self):
        return self.fitness_fx(self.genome)

    def mutate(self,
               num_loc: int,
               magnitude: int):
        locations = np.random.choice(
            range(len(self.genome)), num_loc, False
        )
        for loc in locations:
            value = self.genome[loc]
            new_vals = list(range(value-magnitude, value+magnitude+1))
            new_val = np.random.choice(new_vals)
            if new_val > self.max_gene_val:
                new_val = new_val % self.max_gene_val
            elif new_val < 0:
                new_val = self.max_gene_val + new_val
            self.genome[loc] = new_val


def tournament(fighters, num_winners, p):
    fighter_fitness = [x.fitness() for x in fighters]
    fighters = [x for _, x in sorted(zip(fighter_fitness, fighters),
                                     key=lambda f: f[0],
                                     reverse=True)]
    winners = list()
    while True:
        for i in range(len(fighters)):
            if np.random.uniform() < p * pow((1 - p), i):
                winners.append(fighters[i])
            if len(winners) == num_winners:
                break
        if len(winners) == num_winners:
            break
    return winners


def find_solution(ls: Landscape,
                  ind: Individual = Individual,
                  fitness_fx: callable = None,
                  generations: int = 100,
                  pop_size: int = 20,
                  num_parents: int = 10,
                  tourn_size: int = 4,
                  tourn_winners: int = 2,
                  tourn_p: float = 0.9,
                  exploit_explore: float = 0.1):
    """

    :param ls:
    :param ind:
    :param fitness_fx:
    :param generations:
    :param pop_size:
    :param num_parents:
    :param tourn_size:
    :param tourn_winners:
    :param tourn_p:
    :param exploit_explore:
    :return:
    """

    mutate_locs = int(ls.n * exploit_explore)
    mutate_magnitude = int(0.5 * ls.max_gene_val * exploit_explore)
    if mutate_locs < 1 or mutate_magnitude < 1:
        raise ValueError("exploit_explore value too small for mutation to "
                         "take place. Solutions will not evolve.")

    fitness_over_time = list()

    population = [ind(ls.get_fitness, ls.n, ls.max_gene_val)
                  for _ in range(pop_size)]
    pop_fitness = [x.fitness() for x in population]
    fitness_over_time.append(max(pop_fitness))
    best_soln = population[pop_fitness.index(max(pop_fitness))]
    gen_best_soln = 0

    for generation in range(1, generations + 1):

        adults = list()
        while len(adults) < num_parents:
            fighters = np.random.choice(population, tourn_size, False)
            adults += tournament(fighters, tourn_winners, tourn_p)

        children = list()
        for _ in range(pop_size - num_parents):
            children.append(copy.deepcopy(np.random.choice(adults)))
            children[-1].mutate(mutate_locs, mutate_magnitude)

        population = adults + children

        pop_fitness = [x.fitness() for x in population]
        best_fitness = max(pop_fitness)
        if best_fitness > best_soln.fitness():
            best_soln = population[pop_fitness.index(best_fitness)]
            gen_best_soln = generation
            fitness_over_time.append(best_fitness)
        else:
            fitness_over_time.append(fitness_over_time[-1])

    return fitness_over_time, best_soln, gen_best_soln


if __name__ == '__main__':
    main()

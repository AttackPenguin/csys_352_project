from __future__ import annotations

import concurrent
import copy
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import product
import threading

from multiprocessing import set_start_method

set_start_method('spawn', force=True)

from multiprocessing import Process, Queue
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import discrete_landscape as dl
import continuous_landscape as cl


def main():
    dttm_start = pd.Timestamp.now()

    # fitnesses = list()
    # for _ in range(100):
    #     landscape = dl.Landscape(20, 0, 20)
    #     fitness_over_time, best_soln, gen_best_soln = \
    #         dl.find_solution(landscape, generations=1000)
    #     fitnesses.append(fitness_over_time)
    # fitness_over_time = np.mean(fitnesses, axis=0)
    # plt.plot(list(range(len(fitness_over_time))), fitness_over_time,
    #          label='k=0')
    #
    # print(f"Elapsed Time {pd.Timestamp.now()-dttm_start}")
    #
    # dttm_start = pd.Timestamp.now()

    fitnesses, solutions, generations = run_solution_set(
        landscape=dl.Landscape,
        ls_kwargs={
            'n': 20,
            'k': 0,
            'max_gene_val': 20
        },
        find_solution=dl.find_solution,
        fs_kwargs={
            'generations': 1000
        },
        iterations=100
    )
    fitness_over_time = np.mean(fitnesses, axis=0)
    plt.plot(list(range(len(fitness_over_time))), fitness_over_time,
             label='k=0')

    print(f"Elapsed Time {pd.Timestamp.now() - dttm_start}")


def find_solution_wrapper(q: Queue,
                          landscape: dl.Landscape | cl.Landscape,
                          ls_kwargs: dict[str, Any],
                          find_solution: callable,
                          fs_kwargs: dict[str, Any]):
    landscape = landscape(**ls_kwargs)
    fitness, solution, generation = find_solution(landscape, **fs_kwargs)
    q.put([fitness, solution, generation])


def run_solution_set(landscape: dl.Landscape | cl.Landscape,
                     ls_kwargs: dict[str, Any],
                     find_solution: callable,
                     fs_kwargs: dict[str, Any],
                     iterations: int,
                     num_threads: int = 10):

    fitnesses = list()
    solutions = list()
    generations = list()
    q = Queue()

    fsw_kwargs = {
        'q': q,
        'landscape': landscape,
        'ls_kwargs': ls_kwargs,
        'find_solution': find_solution,
        'fs_kwargs': fs_kwargs,
    }

    # jobs = [find_solution_wrapper(**fsw_kwargs) for _ in range(iterations)]

    completed = 0
    while iterations > completed:
        processes = list()
        if iterations - completed >= num_threads:
            loc_iter = num_threads
        else:
            loc_iter = iterations - completed
        for _ in range(loc_iter):
            processes.append(Process(target=find_solution_wrapper,
                                     kwargs=fsw_kwargs))
        for process in processes:
            process.start()
        for process in processes:
            fitness, solution, generation = q.get()
            fitnesses.append(fitness)
            solutions.append(solution)
            generations.append(generation)
            process.join()
        completed += len(processes)
        print(f"{completed} iterations completed...")

    return fitnesses, solutions, generations


if __name__ == '__main__':
    main()

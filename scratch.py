import os

import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

sns.set_theme(context='paper')

directory = "/home/denis/Desktop/CSYS 352 - Evolutionary Computation/" \
            "Project Proposal"

################################################################################




################################################################################


n = 20
k_vals = [20, 5, 1]
data = list()
for k in k_vals:
    soln_results = list()
    steps_results = list()
    fitness_results = list()
    for _ in range(500):
        landscape = Landscape(n, k)
        soln, fitness, steps = find_nearest_optima(landscape)
        soln_results.append(soln)
        steps_results.append(steps)
        fitness_results.append(fitness)
    data.append(steps_results)


def multiple_histogram(ax: plt.Axes,
                       data: list[list],
                       labels: list):
    max_steps = 0
    for values in data:
        max_value = max(values)
        if max_value > max_steps:
            max_steps = max_value

    for i in range(len(data)):
        ax.hist(data[i],
                bins=list(range(max_steps+2)),
                density=True,
                align='left',
                label=labels[i],
                alpha=0.6)


fig: plt.Figure = plt.figure(figsize=(6, 4), dpi=300)
ax: plt.Axes = fig.add_subplot()
multiple_histogram(ax,
                   data,
                   k_vals)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_title(f'Distribution of Distance to Local Optima\n'
             f'for n=20 and Different Values for k',
             fontsize=14)
ax.set_xlabel('Steps to Optima', fontsize=12)
ax.legend(title='Value of k', fontsize=12)

fig.savefig(os.path.join(directory, 'Distribution of Distance to Local '
                                    'Optima.png'))

################################################################################

n = 20
k_vals = list(range(1, 21))
data = list()
for k in k_vals:
    soln_results = list()
    steps_results = list()
    fitness_results = list()
    for _ in range(500):
        landscape = Landscape(n, k)
        soln, fitness, steps = find_nearest_optima(landscape)
        soln_results.append(soln)
        steps_results.append(steps)
        fitness_results.append(fitness)
    data.append(steps_results)

fig: plt.Figure = plt.figure(figsize=(6, 4), dpi=300)
ax: plt.Axes = fig.add_subplot()
ax.plot(k_vals, [np.mean(x) for x in data])
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator((MaxNLocator(integer=True)))
ax.set_title(f'Mean Distance to Local Optima\n'
             f'for n=20 as a Function of k', fontsize=14)
ax.set_xlabel('Value of k', fontsize=12)
ax.set_ylabel('Steps', fontsize=12)
fig.savefig(os.path.join(directory, "Mean Distance to Local Optima"))



import matplotlib.pyplot as plt
import plotting
import numpy as np
from .result_handler import ResultHandler

def overview_barplot(result_handler: ResultHandler):
    other_colors = ['lightblue', 'darkgray']

    categories = ['betti_0', 'betti_1', 'betti_2', 'name', 'orientability']
    values, errors = result_handler.get_task_means()

    plotting.prepare_for_latex()

    fig, ax = plt.subplots(figsize=(8, 6))

    bar_width = 0.6 
    betti_positions = np.arange(3) 
    other_positions = np.arange(3, 5) + 1  

    for i, pos in enumerate(betti_positions):
        ax.bar(pos, values[i], yerr=errors[i], color='lightcoral', width=bar_width, label=categories[i])

    for i, pos in enumerate(other_positions):
        ax.bar(pos, values[i + 3], yerr=errors[i + 3], color=other_colors[i], width=bar_width, label=categories[i + 3])

    ax.set_xticklabels( ['','betti_0', 'betti_1', 'betti_2', '','name', 'orientability'])
    ax.set_ylabel("Mean Accuracy")
    ax.set_xlabel("Task")
    ax.grid(True, linestyle='--', linewidth=0.5)
    plt.savefig("plot.pdf")
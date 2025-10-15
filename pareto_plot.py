# pareto_plot.py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def plot_pareto_front(pareto_front, save_path="static/img/pareto.png"):
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not pareto_front:
        # No solutions; create empty placeholder
        plt.figure(figsize=(8,6))
        plt.text(0.5, 0.5, 'No solutions', ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()
        return save_path

    costs = [ind.fitness.values[0] for ind in pareto_front]
    waits = [ind.fitness.values[1] for ind in pareto_front]
    plt.figure(figsize=(8,6), facecolor='#0b1b2b')
    ax = plt.subplot(111, facecolor='#0b1b2b')
    ax.scatter(waits, costs, s=100, edgecolors='w', linewidth=0.8)
    ax.set_xlabel("Average Waiting Time (min)", color='white')
    ax.set_ylabel("Operational Cost (₹)", color='white')
    ax.set_title("Pareto Front: Cost vs Waiting Time", color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=plt.gcf().get_facecolor())
    plt.close()
    return save_path

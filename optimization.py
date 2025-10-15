# optimization.py
import random
from deap import base, creator, tools, algorithms
from objectives import calculate_cost, calculate_waiting_time
from simulation import routes

# To avoid creating duplicate creators on repeated imports, use try/except
try:
    creator.FitnessMin
except (AttributeError, Exception):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
try:
    creator.Individual
except (AttributeError, Exception):
    creator.create("Individual", list, fitness=creator.FitnessMin)

def run_optimization(n_pop=50, n_gen=50, freq_min=1, freq_max=6, cxpb=0.7, mutpb=0.2):
    # Toolbox must be recreated each run to respect dynamic frequency bounds
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, freq_min, freq_max)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=len(routes))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return calculate_cost(ind), calculate_waiting_time(ind)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=freq_min, up=freq_max, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    # Initialize population and evaluate
    population = toolbox.population(n=n_pop)
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)

    # Evolution loop
    for gen in range(n_gen):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        population = toolbox.select(population + offspring, k=len(population))

    pareto_front = tools.sortNondominated(population, k=len(population), first_front_only=True)[0]
    # Sort pareto by waiting time (ascending) for display stability
    pareto_front.sort(key=lambda ind: ind.fitness.values[1])
    return pareto_front

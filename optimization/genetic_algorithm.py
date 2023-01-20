import random
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from data.relation_pl_maisons import get_dict_poste_livraisons_maisons
from optimization.schedule import Schedule, Reseau

# %%

poste_livraisons_maisons = get_dict_poste_livraisons_maisons(limite_poste_livraisons=1, limit_maisons_par_pl=100)
schedules = [Schedule(logement, p) for p, logements in poste_livraisons_maisons.items() for logement in logements]

# %%

schedules[0].consommation


# %% [markdown]
# # 1. Initialisation de la population
def init_indivual(poste_livraisons_maisons: dict) -> Reseau:
    schedules = [Schedule(logement, p) for p, logements in poste_livraisons_maisons.items() for logement in logements]
    return Reseau(schedules=schedules)


def init_population(poste_livraisons_maisons: dict, size: int):
    return [init_indivual(poste_livraisons_maisons) for _ in range(size)]


test_population = init_population(poste_livraisons_maisons, 3)

test_population
len(test_population)

# %%

test_population[0].get_global_consommation()


# %% [markdown]
# # 2. Selection des parents

def to_probas(w, inverse=False):
    if inverse:
        w = np.reciprocal(w)
    return np.array(w) / np.sum(w)


def choices_candidates(candidates: list, k, p: List[float]):
    """ Sample k elements (randomly) from candidates with weights """
    # print("p",p,np.sum(p))
    indexes_candidates = list(range(len(candidates)))
    indexes_choosen = []

    while len(indexes_choosen) < k:
        indexes_candidates_who_stay = [i for i in indexes_candidates if i not in indexes_choosen]
        weights_who_stay = [p[i] for i in range(len(p)) if i not in indexes_choosen]
        index = random.choices(indexes_candidates_who_stay, weights=weights_who_stay)[0]
        indexes_choosen.append(index)

    return [candidates[i] for i in indexes_choosen]


# c = [15, 16, 17, 18, 19, 20]
# w = [3, 50, 0.1, 1, 50, 20]
# choices_candidates(c, k=3, p=to_probas(w, inverse=True))


# %%

def get_best(population: List[Reseau]) -> Reseau:
    return sorted(population)[0]


def selection(population: List[Reseau], pct_retain: float = 0.1, pct_selection: float = 0.4) -> List[Reseau]:
    # crerr 2 mode de selection : on garde les pct_retain meilleurs, et on choisie les autres au hasard pondére par le fotness
    population_sorted = sorted(population)

    how_many_retain = int(len(population_sorted) * pct_retain)
    how_many_selection = int(len(population_sorted) * pct_selection)

    selected_individuals = population_sorted[:how_many_retain]
    others_individuals = population_sorted[how_many_retain:]

    w = [ind.get_global_cost() for ind in others_individuals]
    return selected_individuals + choices_candidates(others_individuals, k=how_many_selection,
                                                     p=to_probas(w, inverse=True))


# test_population = selection(population=test_population)
# print(len(test_population))
# test_population
# %%
# test_population[1].print(display_stats=False)


# %% [markdown]
# # 3. Crossover
def crossover(ind1: Reseau, ind2: Reseau) -> Tuple[Reseau]:
    # todo : Croisement en profondeur :
    #  cette méthode consiste à choisir au hasard une profondeur
    #  dans l'arbre de chaque individu parent, puis à échanger
    #  tous les nœuds en dessous de cette profondeur entre les deux individus
    #  pour créer les deux individus enfants.

    schedules1 = ind1.get_leaves_schedules()
    schedules2 = ind2.get_leaves_schedules()
    n_schedules = len(schedules1)
    middle = n_schedules // 2
    new_schedules1 = schedules1[:middle] + schedules2[middle:]
    new_schedules2 = schedules2[:middle] + schedules1[middle:]

    return Reseau(new_schedules1), Reseau(new_schedules2)


#crossover(test_population[0], test_population[1])


# %%
def crossing(population: List[Reseau], n_newborns: int, crossover_rate: float = 0.7) -> List[Reseau]:
    population_sorted = sorted(population)

    how_many_retain = int(len(population_sorted) * crossover_rate)
    future_parent = population_sorted[:how_many_retain]
    newborns = []
    while len(newborns) < n_newborns:
        newborns += [born for born in crossover(*random.choices(future_parent, k=2))]
    print(type(newborns), type(newborns[0]))
    return population + newborns


#test_population = crossing(test_population, n_newborns=50)

# %% [markdown]
# # 4. Mutation

#test_population[0].print(display_stats=False)

#tmp = test_population[0]._mutate_leaves(3)

#test_population[0].print(display_stats=False)


# %%

def mutation(population: List[Reseau], mutation_rate: float = 0.1) -> List[Reseau]:
    for ind in population:
        if random.random() < mutation_rate:
            ind.mutate(n_leaves_mutations=5)


def evolve(population: List[Reseau], num_generation: int, pct_retain: float = 0.1, pct_selection: float = 0.4,
           n_newborns: int = 50, mutation_rate: float = 0.1, crossover_rate: float = 0.7) -> List[Reseau]:
    population = selection(population, pct_retain, pct_selection)
    population = crossing(population, n_newborns, crossover_rate)
    mutation(population, mutation_rate)
    return population


def genetic_algorithm(poste_livraisons_maisons: dict, max_generations: int, population_size: int,
                      mutation_rate: float, crossover_rate: float, n_newborns: int, selection_rate: float,
                      verbose: bool = True):
    print(">> Starting genetic algorithm... ")
    print(f"> Parameters : \n",
          f" * max_generations : {max_generations}\n",
          f" * population_size : {population_size}\n",
          f" * mutation_rate : {mutation_rate}\n",
          f" * crossover_rate : {crossover_rate}\n",
          f" * n_newborns : {n_newborns}\n",
          f" * selection_rate : {selection_rate}\n\n",
          )
    history_best_individuals = []
    population = init_population(poste_livraisons_maisons, population_size)
    for num_generation in range(max_generations):
        population = evolve(population, num_generation=num_generation, pct_retain=selection_rate,
                            pct_selection=selection_rate,
                            n_newborns=n_newborns, mutation_rate=mutation_rate, crossover_rate=crossover_rate)
        history_best_individuals.append(get_best(population))

        if verbose:
            print(
                f"Generation {num_generation + 1} : best_cost={round(history_best_individuals[-1].get_global_cost(), 2)}")

    return get_best(population), history_best_individuals


def plot_history(history_best_individuals: List[Reseau]):
    # Plot the evolution of the best individual
    plt.figure(figsize=(10, 5))
    # subplot 1 : Evolution of the best individual (cost)
    plt.subplot(1, 2, 1)
    plt.plot([ind.get_global_cost() for ind in history_best_individuals])
    plt.title("Evolution of the best individual")
    plt.xlabel("Generation")
    plt.ylabel("Cost")
    # subplot 2 : Evolution of the best individual (consommation)
    plt.subplot(1, 2, 2)
    plt.plot([ind.get_global_consommation() for ind in history_best_individuals])
    plt.title("Evolution of the best individual")
    plt.xlabel("Generation")
    plt.ylabel("Consommation")
    plt.show()


def main():
    rel_poste_livraisons_maisons = get_dict_poste_livraisons_maisons(limite_poste_livraisons=10, limit_maisons_par_pl=3)
    best_reseau, history_best_individuals = genetic_algorithm(rel_poste_livraisons_maisons, max_generations=50,
                                                              population_size=100,
                                                              mutation_rate=0.1, crossover_rate=0.7, n_newborns=500,
                                                              selection_rate=0.2)

    best_reseau.print()
    plot_history(history_best_individuals)


if __name__ == "__main__":
    main()

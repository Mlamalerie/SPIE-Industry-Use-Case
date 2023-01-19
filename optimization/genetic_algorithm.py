import random
from typing import List
from typing import Tuple

import numpy as np

from data.relation_reseau_maisons import get_dict_parents_enfants
from optimization.schedule import Schedule, Reseau

# %%

rel_parents_enfants_maisons = get_dict_parents_enfants(limit_parents=2, limit_child_per_parent=5)
example_schedules_load = [Schedule(logement, p) for p, logements in rel_parents_enfants_maisons.items() for logement in
                          logements]
len(example_schedules_load)

# %%

example_schedules_load[0].genome


# %% [markdown]
# # 1. Initialisation de la population
def init_indivual(parents_enfants: dict) -> Reseau:
    schedules = [Schedule(logement, p) for p, logements in parents_enfants.items() for logement in logements]
    return Reseau(schedules=schedules)


def init_population(parents_enfants: dict, size: int):
    return [init_indivual(parents_enfants) for _ in range(size)]


test_population = init_population(rel_parents_enfants_maisons, 100)

test_population


# %% [markdown]
# # 2. Selection des parents

def to_probas(w, inverse=False):
    if inverse:
        w = [1 / i for i in w]
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


c = [15, 16, 17, 18, 19, 20]
w = [3, 50, 0.1, 1, 50, 20]
choices_candidates(c, k=3, p=to_probas(w, inverse=True))


# %%

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


test_population = selection(population=test_population)
print(len(test_population))
test_population
# %%
test_population[1].print(display_stats=False)


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


crossover(test_population[0], test_population[1])


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


test_population = crossing(test_population, n_newborns=50)

# %% [markdown]
# # 4. Mutation

test_population[0].print(display_stats=False)

tmp = test_population[0]._mutate_leaves(3)

test_population[0].print(display_stats=False)


# %%

def mutation(individual: Reseau, mutation_rate: float = 0.1) -> Reseau:
    if random.random() < mutation_rate:
        individual.mutate()




def evolve(population: List[Reseau], ):
    """Creates the next generation of a given population with the
    given parameters.
    """

    # list of individuals ordered by fitness

    # a portion of the most fit individuals become parents
    # parents = graded[:int(len(graded)*pct_retain))


def genetic_algorithm(parents_enfants: dict, max_generations: int = 1000, population_size: int = 100,
                      mutation_rate: float = 0.1, crossover_rate: float = 0.7, selection_rate: float = 0.2) -> Reseau:
    pass


def main():
    pass


if __name__ == "__main__":
    main()

import random
from typing import List
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from data.relation_pl_maisons import get_dict_poste_livraisons_maisons
from optimization.schedule import Schedule, Reseau


def display_loading_bar(i, n, bar_length=20, text=""):
    percent = float(i) / n
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(
        '2: [0] 1%'.format(arrow + spaces, int(round(percent * 100))),
        text or 'Loading',
        end="\r",
    )


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


def selection(population: List[Reseau], selection_retain_rate: float = 0.1, selection_rate: float = 0.4) -> List[
    Reseau]:
    # crerr 2 mode de selection : on garde les pct_retain meilleurs, et on choisie les autres au hasard pondére par le fotness
    population_sorted = sorted(population)

    how_many_retain = int(len(population_sorted) * selection_retain_rate)
    how_many_selection = int(len(population_sorted) * selection_rate)

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


# crossover(test_population[0], test_population[1])


# %%
def crossing(population: List[Reseau], n_newborns: int, crossover_rate: float = 0.7) -> List[Reseau]:
    population_sorted = sorted(population)

    how_many_retain = int(len(population_sorted) * crossover_rate)
    future_parent = population_sorted[:how_many_retain]
    newborns = []
    while len(newborns) < n_newborns:
        newborns += list(crossover(*random.choices(future_parent, k=2)))
    # print(type(newborns), type(newborns[0]))
    return population + newborns


# test_population = crossing(test_population, n_newborns=50)

# %% [markdown]
# # 4. Mutation

# test_population[0].print(display_stats=False)

# tmp = test_population[0]._mutate_leaves(3)

# test_population[0].print(display_stats=False)


# %%

def mutation(population: List[Reseau], mutation_rate: float = 0.1) -> List[Reseau]:
    for ind in population:
        if random.random() < mutation_rate:
            ind.mutate(n_leaves_mutations=5)


def evolve(population: List[Reseau], selection_retain_rate: float, selection_rate: float, mutation_rate: float,
           crossover_rate: float) -> List[Reseau]:
    population = selection(population, selection_retain_rate=selection_retain_rate, selection_rate=selection_rate)
    population = crossing(population, n_newborns=len(population), crossover_rate=crossover_rate)
    mutation(population, mutation_rate)
    return population


def get_stats_from_population(population: List[Reseau]):
    costs = [ind.get_global_cost() for ind in population]
    # return min max mean std
    return np.mean(costs), np.std(costs)


def genetic_algorithm(poste_livraisons_maisons: dict, max_generations: int, population_size: int,
                      mutation_rate: float, crossover_rate: float, selection_retain_rate: float,
                      selection_rate: float, verbose: bool = True, eps_std_stop: float = 0.01,
                      stop_after_n_gen: int = 10) -> Tuple[Reseau, List[float]]:
    if verbose:
        print(">> Starting genetic algorithm... ")
        print(f"> Parameters : \n",
              f" * max_generations : {max_generations}\n",
              f" * population_size : {population_size}\n",
              f" * selection_rate : {selection_rate}\n",
              f" * selection_retain_rate : {selection_retain_rate}\n",
              f" * crossover_rate : {crossover_rate}\n",
              f" * mutation_rate : {mutation_rate}\n\n",
              )
    history_best_individuals = []
    history_mean_costs = []
    history_std_costs = []
    population = init_population(poste_livraisons_maisons, population_size)
    for num_generation in range(max_generations):
        population = evolve(population, selection_retain_rate=selection_retain_rate, selection_rate=selection_rate,
                            mutation_rate=mutation_rate, crossover_rate=crossover_rate)
        history_best_individuals.append(get_best(population))
        # get stats
        mean_cost, std_cost = get_stats_from_population(population)
        history_mean_costs.append(mean_cost)
        history_std_costs.append(std_cost)
        if std_cost < eps_std_stop:
            # stop because the std of all population cost is too low
            print(f">> Stopping genetic algorithm because std of all population cost is too low : {std_cost}")
            break
        # the last 5 generations have the same best cost

        if len(history_best_individuals) > stop_after_n_gen and np.all(
                [history_best_individuals[-1] == history_best_individuals[-i] for i in range(2, stop_after_n_gen + 1)]):
            print(
                f">> Stopping genetic algorithm because the last 5 generations have the same best cost : {history_best_individuals[-1]}")
            break

        if verbose:
            print(
                f" # Generation {num_generation + 1} (pop={len(population)}) : best_cost={round(history_best_individuals[-1].get_global_cost(), 2)}")
    if verbose:
        print(">> Genetic algorithm finished !")
    return get_best(population), num_generation, history_best_individuals, history_mean_costs, history_std_costs


def plot_history(history_best_individuals: List[Reseau], history_mean_costs: List[float],
                 history_std_costs: List[float]):
    # Plot the evolution of the best individual
    # subplot 1 : Evolution of the best individual (cost)
    fig_1, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig_1.tight_layout(pad=4.0)

    axs[0].plot([ind.get_global_cost() for ind in history_best_individuals], color='red')
    axs[0].set_title("Evolution of the best individual (cost)", size=10)
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Cost")
    # subplot 2 : Evolution of the best individual (consommation)

    axs[1].plot([ind.get_global_consommation() for ind in history_best_individuals], color='orange')
    axs[1].set_title("Evolution of the best individual (consommation)", size=10)
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Consommation")

    # new figure : Evolution of the mean and std of the population
    # subplot 1 : mean
    fig_2, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig_2.tight_layout(pad=4.0)
    axs[0].plot(history_mean_costs, color="teal")
    axs[0].set_title("Evolution of the mean cost of the population", size=10)
    axs[0].set_xlabel("Generation")
    axs[0].set_ylabel("Mean cost")
    # subplot 2 : std
    # plot std and add a line at 0
    axs[1].plot(history_std_costs, color='blue')
    axs[1].axhline(y=0, color='black', linestyle='-')
    axs[1].set_title("Evolution of the std cost of the population", size=10)
    axs[1].set_xlabel("Generation")
    axs[1].set_ylabel("Std cost")

    return fig_1, fig_2

@st.cache(suppress_st_warning=True)
def launch_optimization(mode="global", limite_poste_livraisons: int = 10, limit_maisons_par_pl: int = 50,
                        max_generations: int = 100, population_size: int = 100, mutation_rate: float = 0.1,
                        crossover_rate: float = 0.7, selection_retain_rate: float = 0.1, selection_rate: float = 0.4,
                        plot_with_streamlit=False, display_plot=True, verbose=True, f_print=print):
    poste_livraisons_maisons = get_dict_poste_livraisons_maisons(limite_poste_livraisons=limite_poste_livraisons,
                                                                 limit_maisons_par_pl=limit_maisons_par_pl)

    # Optimisation globale
    if mode == "global":
        f_print(
            f">> Starting global optimization with {len(poste_livraisons_maisons)} PL and {sum(len(maisons) for maisons in poste_livraisons_maisons.values())} maisons"
        )
        best_reseau, num_generation, history_best_individuals, history_mean_costs, history_std_costs = genetic_algorithm(
            poste_livraisons_maisons, max_generations=max_generations, population_size=population_size,
            mutation_rate=mutation_rate, crossover_rate=crossover_rate, selection_retain_rate=selection_retain_rate,
            selection_rate=selection_rate, verbose=verbose)
        f_print(">> Best individual :")
        best_reseau.print()
        if display_plot:
            f_print(">> Plotting history...")
            fig_1, fig_2 = plot_history(history_best_individuals, history_mean_costs, history_std_costs)
            if plot_with_streamlit:
                st.pyplot(fig_1)
                st.pyplot(fig_2)
            else:
                fig_1.show()
                fig_2.show()

        return {"global_best_reseau": best_reseau}
    elif mode == "local":
        best_reseaux_pl = {}
        # for each poste de livraison, optimize the reseau
        for poste_livraison in poste_livraisons_maisons.keys():
            f_print(f">> Starting optimization for poste de livraison {poste_livraison}...")
            best_reseau, num_generation, history_best_individuals, history_mean_costs, history_std_costs = genetic_algorithm(
                {poste_livraison: poste_livraisons_maisons[poste_livraison]}, max_generations=max_generations,
                population_size=population_size,
                mutation_rate=mutation_rate, crossover_rate=crossover_rate, selection_retain_rate=selection_retain_rate,
                selection_rate=selection_rate, verbose=verbose)
            f_print(">> Best individual :")
            best_reseau.print()
            if display_plot:
                f_print(">> Plotting history...")
                fig_1, fig_2 = plot_history(history_best_individuals, history_mean_costs, history_std_costs)
                if plot_with_streamlit:
                    st.pyplot(fig_1)
                    st.pyplot(fig_2)
                else:
                    fig_1.show()
                    fig_2.show()
            best_reseaux_pl[poste_livraison] = best_reseau

        return {"local_best_reseaux": best_reseaux_pl}  # return a dict of best reseaux for each poste de livraison


def main():
    LUNCH_MODE = "global"  # "global" or "local"
    result = launch_optimization(LUNCH_MODE,
                                 limite_poste_livraisons=3, limit_maisons_par_pl=10,
                                 max_generations=50, population_size=1000,
                                 mutation_rate=0.1, crossover_rate=0.7, selection_retain_rate=0.1, selection_rate=0.4,
                                 plot_with_streamlit=False, verbose=True)



if __name__ == "__main__":
    main()

import random
from typing import List

from data.equipements import EquipementsDataManager
from optimization.schedule import Schedule, Reseau

# %%

s = Schedule("2", '5')
print(s.get_cost())


# %%



# %%
def init_schedule(logement_name, parent_name, equipement_manager: EquipementsDataManager) -> Schedule:
    """ Initialize a random schedule for a logement """

    return Schedule(logement_name, parent_name)


def init_solution(parents_enfants: dict) -> Reseau:
    pass


def sample(candidates, k, weights):
    result = []
    while len(result) < k:
        choosen = random.choices(candidates, weights=weights)[0]
        if choosen not in result:
            result.append(choosen)

    return result


def crossover(solution1: Reseau, solution2: Reseau) -> List[Reseau]:
    pass


def mutation(solution: Reseau) -> Reseau:
    pass




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

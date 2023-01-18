from typing import List

from src.data.datamanagers import EquipementsDataManager, LimitePuissanceDataManager
from src.optimization.modeles import Schedule, Reseau

HEURES_CREUSES = ("00:00", "07:00")

# %%


s = Schedule("2", '5')
print(s.get_cost())


# %%

def logement_name_2_surface(logement_name):
    return int(logement_name.split("_")[0][1:])


def verify_schedule_constraint(schedule: Schedule, lpm: LimitePuissanceDataManager):
    schedule.logement_name  # logement name
    surface = logement_name_2_surface(schedule.logement_name)  # todo : ? stoquer surface dan class
    kWa_limit = lpm.get_limit_power(surface)["kWa"]  # limite de puissance
    cost = schedule.get_cost()
    if cost > kWa_limit:
        return False, cost - kWa_limit

    return True, 0


# %%
def init_schedule(logement_name, parent_name, equipement_manager: EquipementsDataManager) -> Schedule:
    """ Initialize a random schedule for a logement """

    return Schedule(logement_name, parent_name)


def init_solution(parents_enfants: dict) -> Reseau:
    pass


def verify_solution_respect_constraint(solution: Reseau):
    pass


def crossover(solution1: Reseau, solution2: Reseau) -> List[Reseau]:
    pass


def mutation(solution: Reseau) -> Reseau:
    pass


def selection(population: List[Reseau], pct_retain: float = 0.2) -> List[Reseau]:
    # crerr 2 mode de selection : on garde les pct_retain meilleurs, et on choisie les autres au hasard pondére par le fotness
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

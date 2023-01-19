import random
import re
from typing import List

import numpy as np
import pandas as pd
from bigtree import Node, print_tree, dict_to_tree
from bigtree import find_name as tree_find_name

from data.equipements import EquipementsDataManager
from data.limites_puissance import LimitesPuissanceDataManager
from data.reseau_maisons import get_dict_parents_enfants

HEURES_CREUSES = ("20:00", "06:00")
HEURES_DEJEUNER = ("11:00", "13:00")
HEURES_DINER = ("18:00", "20:00")


# %%


def hour_2_index(hour: str) -> int:
    """Convert hour to index in vector"""
    return int(hour.split(":")[0])


def put_value_randomly_on_vector(vector, value: float, how_many: int = 1, index_range: tuple = None,
                                 sequence: bool = False):
    """Put several times, randomly, the same value in different index of a vector
    :param vector: vector to modify
    :param value: value to put in vector
    :param how_many: how many times to put the value
    :param index_range: range of index where to put the value [index_start, index_end]
    """

    len_vector = len(vector)

    indexes = None
    if index_range is None:
        index_range = (0, len_vector - 1)
    elif max(index_range) >= len_vector or min(index_range) < 0:
        raise ValueError(f"index_range {index_range} is out of range of vector {len_vector}")

    if index_range[0] < index_range[1]:

        len_range = index_range[1] - index_range[0] + 1
        how_many = min(how_many, len_range)
        indexes_range_list = list(range(index_range[0], index_range[1] + 1))
        if not sequence:
            indexes = random.sample(indexes_range_list, how_many)
        else:
            index_choose = random.choice(indexes_range_list[:-how_many + 1] if how_many > 1 else indexes_range_list)
            j = indexes_range_list.index(index_choose)
            indexes = indexes_range_list[j: j + how_many]
            # indexes = [i % len_vector for i in indexes]
    elif index_range[0] > index_range[1]:
        # exemple : 23:00 -> 07:00 [23 -> 7]
        len_range = index_range[1] - index_range[0] + len_vector + 1
        how_many = min(how_many, len_range)
        indexes_range_list = list(range(index_range[0], len_vector)) + list(range(0, index_range[1] + 1))
        if not sequence:
            indexes = random.sample(indexes_range_list, how_many)
        else:
            index_choose = random.choice(indexes_range_list[:-how_many + 1] if how_many > 1 else indexes_range_list)
            j = indexes_range_list.index(index_choose)
            indexes = indexes_range_list[j: j + how_many]
    else:
        indexes = [index_range[0]]

    # print("index_range", index_range, indexes_range_list, "[:-how_many]", indexes_range_list[:-how_many + 1],
    #      "len_range", len_range, "how_many", how_many)
    # print("indexes", indexes)

    for i in indexes:
        vector[i] = value
    return vector


z = np.zeros((3, 24), dtype=np.float32)
z[2] = put_value_randomly_on_vector(z[2], value=0.2, how_many=6, sequence=True)
z


# %%
class Schedule:
    eqm = EquipementsDataManager()
    lpm = LimitesPuissanceDataManager()

    def __init__(self, logement_name, parent_name):
        self.logement_name = logement_name
        self.parent_name = parent_name
        self.logement_equipements: list = Schedule.eqm.get_equipements_by_logement_name(self.logement_name)
        self.genome: np.array = np.zeros((len(Schedule.eqm.equipements_names), 24), dtype=float)
        self.init_random_genome()
        self.cost: float = 100.0 + round(random.random(), 2)
        self.consommation: float = 0.0
        self.evaluate_consommation()

        self.mutation_rate = 0.1  # 10% chance of mutation
        # self.decrease_for_constraint_violation = 0.25  # 25% decrease in cost if constraint is violated

    def init_random_genome(self):
        # todo : tc and other
        # for each equipement in the logement
        # print("equipements_list", self.logement_equipements)
        for logement_equipement_name in self.logement_equipements:
            index_genome_matrix = Schedule.eqm.get_index_equipement_by_name(logement_equipement_name)
            # get caracteristics of equipement

            if caracteristiques := Schedule.eqm.caracteristiques_equipements.get(logement_equipement_name):
                t_cycle = caracteristiques.get("tps_cycle")
                t_cycle = round(t_cycle, 0) if t_cycle > 1 else 1.0  # d'après jenna
                puissance = caracteristiques.get("puissance")
                sequensable = caracteristiques.get("sequensable")

                hr_debut = caracteristiques.get("hr_debut")

                how_many_to_put = int(t_cycle)
                value_to_put = round(puissance / t_cycle, 1)

                horaire_range = None
                if hr_debut > 0:

                    horaire_range = (hour_2_index(HEURES_CREUSES[0]), hour_2_index(HEURES_CREUSES[1]))
                    for _ in range(hr_debut):
                        self.genome[index_genome_matrix] = put_value_randomly_on_vector(
                            self.genome[index_genome_matrix],
                            value=value_to_put,
                            how_many=how_many_to_put,
                            index_range=horaire_range,
                            sequence=sequensable)
                else:
                    self.genome[index_genome_matrix] = put_value_randomly_on_vector(self.genome[index_genome_matrix],
                                                                                    value=value_to_put,
                                                                                    how_many=how_many_to_put,
                                                                                    index_range=horaire_range,
                                                                                    sequence=sequensable)

                # if logement_equipement_name == "Mc2-CE":
                #    print("Mc2-CE", self.genome[index_genome_matrix])

            else:
                raise ValueError(f"caracteristiques not found for equipement")

    def get_surface(self) -> int:
        return int(self.logement_name.split("-")[0][1:])

    def is_respect_constraint(self, puissance_limite: float = None) -> bool:
        puissance_limite = Schedule.lpm.get_limites(self.get_surface())[
            "kVA"] if puissance_limite is None else puissance_limite
        sum_vect = np.sum(self.genome, axis=0)  # array avec sommes des puissance

        return np.all(sum_vect <= puissance_limite), np.sum(sum_vect > puissance_limite)

    def evaluate_consommation(self):
        self.consommation = np.sum(self.genome)

    def get_cost(self):
        cost = 0.0

        ok, nb_violation = self.is_respect_constraint()
        if ok:
            return self.cost
        else:
            return self.cost * nb_violation

    def get_consommation(self):
        return self.consommation

    def get_parent_name(self):
        return self.parent_name

    def get_logement_name(self):
        return self.logement_name

    def mutate(self):
        if random.random() < self.mutation_rate:
            print("*")


def vizualise_schedule(schedule: Schedule):
    hours_str = [f"{i:02d}:00" for i in range(24)]
    # to pandas
    df = pd.DataFrame(schedule.genome, columns=hours_str, index=Schedule.eqm.equipements_names)
    return df


# logements = list(Schedule.eqm.equipements_list_par_logements.keys())
# ss = Schedule(logements[1], "PL1111")

# print("done")


# %%

def level_up_pl(pl_id: str) -> str:
    # keep digits
    id = re.sub(r"[^0-9]", "", pl_id)
    # print(id)
    depth = len(id)
    return 'PS' if depth <= 1 else pl_id[:-1]


def get_all_ancestors(pl_id: str, sort_desc=True) -> list:
    ancestors = []
    while pl_id != 'PS':
        pl_id = level_up_pl(pl_id)
        ancestors.append(pl_id)
    return ancestors[::-1] if sort_desc else ancestors


# print(get_all_ancestors("PL122"))


# %%

class Reseau():
    def __init__(self, schedules: List[Schedule]):
        self.schedules = schedules

        self.tree = None
        self.global_cost: float = 0.0
        # self.base_leaves_paths_list = None
        self.base_leaves_paths_dict = None

        # self.parents_enfants = get_dict_parents_enfants(limit_parents=limit_parents,
        #                                                limit_child_per_parent=limit_child_per_parent)
        self.construct_tree()
        self.load_global_cost()

    def logement_name_to_node_name(self, parent_name, logement_name: str) -> str:
        return "/".join(
            get_all_ancestors(parent_name) + [f"{parent_name}/{logement_name}"]
        )

    def sum_leafs(self, node: Node) -> float:
        if node.is_leaf:
            return node.schedule.cost
        else:
            return sum(self.sum_leafs(child) for child in node.children)

    def load_global_cost(self) -> None:
        self.global_cost = self.sum_leafs(self.tree)

    def get_global_cost(self) -> float:
        return self.global_cost

    def __lt__(self, other):
        return self.global_cost < other.global_cost

    def get_cost_by_parent(self, parent_name: str) -> float:
        node = tree_find_name(self.tree, parent_name)
        if node is None:
            raise ValueError(f"Parent {parent_name} not found")
        return self.sum_leafs(node)

    def get_basic_stats_leafs_cost(self) -> float:
        costs = np.array([leaf.schedule.cost for leaf in self.tree.leaves])
        return np.mean(costs), np.std(costs), np.min(costs), np.max(costs)

    def construct_tree(self, verbose=False):

        # self.path_dict = {k: {"cost": round(random(), 1)} for k in result}  # todo
        if len(self.schedules) == 0:
            raise ValueError("No schedules in Reseau")

        self.base_leaves_paths_dict = {}
        # for each schedule
        for schedule in self.schedules:
            leaf_path = self.logement_name_to_node_name(schedule.parent_name, schedule.logement_name)
            self.base_leaves_paths_dict[leaf_path] = {"schedule": schedule}

        self.tree = dict_to_tree(self.base_leaves_paths_dict)

        # bring up cost to parents
        # for node in self.tree.traverse("postorder"):
        if verbose:
            print("tree construction done.")

    def print(self, display_stats=True):
        if self.tree is None:
            raise ValueError("Tree is not constructed yet")

        if display_stats and self.schedules:
            self._print_stats()
        print_tree(self.tree, attr_list=["cost"])
        print("-" * 50)

    # TODO Rename this here and in `print`
    def _print_stats(self):
        n_leaves = len(list(self.tree.leaves))
        conso = self.get_global_cost()
        print("-" * 1, "GLOBAL STATS", "-" * 35)
        print("* Cost accumulés:", conso, "\n")
        print("-" * 1, f"LEAFS STATS ({n_leaves} logements)", "-" * 22)
        mean_, std_, min_, max_ = self.get_basic_stats_leafs_cost()
        print("* Cost moyenne:", mean_)
        print("* Cost std:", round(std_))
        print("* Cost min:", min_)
        print("* Cost max:", max_)
        print("-" * 50)

    def get_schedule_by_leaf_path(self, leaf_path: str) -> Schedule:
        return self.base_leaves_paths_dict[leaf_path]["schedule"]

    def set_schedules(self, schedules: List[Schedule]):
        self.schedules = schedules
        self.construct_tree()

    def get_schedules(self) -> List[Schedule]:
        return self.schedules


if __name__ == "__main__":
    # reseau = Reseau()
    # reseau.print()
    parents_enfants = get_dict_parents_enfants(2, 5)
    # example_schedules = [Schedule(logement, p) for p, logements in parents_enfants.items() for logement in logements]
    # reseau = Reseau(schedules=example_schedules)

    # reseau.print()

    print("done")

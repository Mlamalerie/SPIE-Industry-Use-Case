import random
import re
from typing import List

import numpy as np
from bigtree import Node, print_tree, dict_to_tree, list_to_tree
from bigtree import find_name as tree_find_name

from src.data.datamanagers import get_dict_parents_enfants, EquipementsDataManager, LimitePuissanceDataManager


# %%

def put_values_on_vector(vector, value: float, how_many: int = 1, sequence: bool = False):
    """Put several times, randomly, the same value in different index of a vector"""
    len_vector = len(vector)
    how_many = min(how_many, len_vector)
    if not sequence:
        indexes = random.sample(range(len_vector), how_many)
    else:
        i = random.randint(0, len_vector - 1)
        indexes = list(range(i, i + how_many))
        indexes = [i % len_vector for i in indexes]

    for i in indexes:
        vector[i] = value
    return vector


# np zeros float
z = np.zeros(10, dtype=float)
put_values_on_vector(z, 1.2, 6, sequence=True)


# %%
class Schedule:
    eqm = EquipementsDataManager()
    lpm = LimitePuissanceDataManager()

    def __init__(self, logement_name, parent_name):
        self.logement_name = logement_name
        self.parent_name = parent_name
        self.logement_equipements: list = Schedule.eqm.get_equipements_by_logement_name(self.logement_name)
        self.genome: np.array = np.zeros((len(Schedule.eqm.equipements_names), 24))
        self.init_random_genome()
        self.cost: float = 0.0
        self.consommation: float = 0.0
        self.evaluate_consommation()

        # self.mutation_rate = 0.1  # 10% chance of mutation
        # self.decrease_for_constraint_violation = 0.25  # 25% decrease in cost if constraint is violated

    def init_random_genome(self):
        # for each equipement in the logement

        print("equipements_list", self.logement_equipements)
        for logement_equipement_name in self.logement_equipements:
            index_genome_matrix = Schedule.eqm.get_index_equipement_by_name(logement_equipement_name)
            # get caracteristics of equipement
            if caracteristiques := Schedule.eqm.caracteristiques_equipements.get(logement_equipement_name):
                t_cycle = caracteristiques.get("tps_cycle")
                sequensable = caracteristiques.get("sequensable")
                how_many = random.randint(1, self.genome.shape[1])
                print(logement_equipement_name, "how_many", how_many, "t_cycle", t_cycle, "sequensable", sequensable)
                put_values_on_vector(self.genome[index_genome_matrix], value=t_cycle,
                                     how_many=how_many, sequence=sequensable)
            else:
                raise ValueError(f"caracteristiques not found for equipement {logement_equipement_name}")

    def is_respect_constraint(self):
        Schedule.lpm.get_limites(self.get_surface())["kVA"]  # KW, or kVA

        for logement_equipement_name in self.logement_equipements:
            index_genome_matrix = Schedule.eqm.get_index_equipement_by_name(logement_equipement_name)

            # sum column

            return True

    def get_surface(self) -> int:
        return int(self.logement_name.split("-")[0][1:])

    def evaluate_consommation(self):
        sum_matrix = np.sum(self.genome, axis=1)  # array avec sommes des temps
        print("1")

    def get_cost(self):
        cost = 0.0
        if self.is_respect_constraint():
            return self.cost
        else:
            return self.cost * self.decrease_for_constraint_violation

    def get_parent_name(self):
        return self.parent_name

    def get_logement_name(self):
        return self.logement_name

    def mutate(self):
        if random.random() < self.mutation_rate:
            print("*")


s = Schedule("A100-3-100", "PL1111")
s.is_respect_constraint()
print("done")


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


print(get_all_ancestors("PL122"))


# %%

class Reseau():
    def __init__(self, limit_parents: int = None, limit_child_per_parent: int = None, schedules: List[Schedule] = None):
        self.tree = None
        self.schedules = schedules
        self.base_leaves_paths_list = None
        self.base_leaves_paths_dict = None

        self.parents_enfants = get_dict_parents_enfants(limit_parents=limit_parents,
                                                        limit_child_per_parent=limit_child_per_parent)
        self.construct_tree()

    def logement_name_to_node_name(self, parent_name, logement_name: str) -> str:
        return "/".join(
            get_all_ancestors(parent_name) + [f"{parent_name}/{logement_name}"]
        )

    def sum_leafs(self, node: Node) -> float:
        if node.is_leaf:
            return node.cost
        else:
            return sum(self.sum_leafs(child) for child in node.children)

    def get_cost_global(self) -> float:
        return self.sum_leafs(self.tree)

    def get_cost_by_parent(self, parent_name: str) -> float:
        node = tree_find_name(self.tree, parent_name)
        if node is None:
            raise ValueError(f"Parent {parent_name} not found")
        return self.sum_leafs(node)

    def get_basic_stats_leafs_cost(self) -> float:
        costs = np.array([leaf.cost for leaf in self.tree.leaves])
        return np.mean(costs), np.std(costs), np.min(costs), np.max(costs)

    def construct_tree(self):

        # self.path_dict = {k: {"cost": round(random(), 1)} for k in result}  # todo
        if not self.schedules:
            leaves_paths_list = []
            for parent_name, children in self.parents_enfants.items():
                leaves_paths_list.extend(
                    self.logement_name_to_node_name(parent_name, child)
                    for child in children
                )
            self.base_leaves_paths_list = leaves_paths_list
            self.tree = list_to_tree(self.base_leaves_paths_list)
        else:
            schedules_leaves_paths_list = []
            self.base_leaves_paths_dict = {}
            # for each schedule
            for schedule in self.schedules:
                leaf_path = self.logement_name_to_node_name(schedule.parent_name, schedule.logement_name)
                schedules_leaves_paths_list.append(leaf_path)
                self.base_leaves_paths_dict[leaf_path] = {"cost": schedule.get_cost()}

            self.tree = dict_to_tree(self.base_leaves_paths_dict)

        # bring up cost to parents
        # for node in self.tree.traverse("postorder"):
        print("The tree has been constructed.")

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
        conso = self.get_cost_global()
        print("-" * 1, "GLOBAL STATS", "-" * 35)
        print("* Cost accumulés:", conso, "\n")
        print("-" * 1, f"LEAFS STATS ({n_leaves} logements)", "-" * 22)
        mean_, std_, min_, max_ = self.get_basic_stats_leafs_cost()
        print("* Cost moyenne:", mean_)
        print("* Cost std:", round(std_))
        print("* Cost min:", min_)
        print("* Cost max:", max_)
        print("-" * 50)


if __name__ == "__main__":
    reseau = Reseau()
    reseau.print()
    parents_enfants = get_dict_parents_enfants(2, 5)
    example_schedules = [Schedule(logement, p) for p, logements in parents_enfants.items() for logement in logements]
    reseau = Reseau(schedules=example_schedules)

    reseau.print()

    print("done")

import re
from typing import List

import numpy as np
from bigtree import Node, print_tree, dict_to_tree, list_to_tree
from bigtree import find_name as tree_find_name

from src.data.datamanagers import get_dict_parents_enfants
from src.optimization.genetic_algorithm import Schedule


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


parents_enfants = get_dict_parents_enfants(2, 5)
example_schedules = [Schedule(logement, p) for p, logements in parents_enfants.items() for logement in logements]
reseau = Reseau(schedules=example_schedules)

reseau.print()

print("done")

print("#")

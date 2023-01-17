import re
from random import random

import numpy as np
from bigtree import Node, print_tree, dict_to_tree
from bigtree import find_name as tree_find_name

from src.data.datamanagers import get_dict_parents_enfants


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
    return ancestors if not sort_desc else ancestors[::-1]


print(get_all_ancestors("PL122"))


# %%

class Reseau():
    def __init__(self, limit_parents: int = None, limit_child_per_parent: int = None, path_logements_dict=None):
        self.tree = None
        self.parents_enfants = get_dict_parents_enfants(heads=limit_parents)
        self.construct_tree(limit_child_per_parent=limit_child_per_parent)

    def logement_name_to_node_name(self, parent_name, logement_name: str) -> str:
        node_name = "/".join(get_all_ancestors(parent_name) + [f"{parent_name}/{logement_name}"])
        return node_name

    def sum_leafs(self, node: Node) -> float:
        if node.is_leaf:
            return node.consommation
        else:
            return sum(self.sum_leafs(child) for child in node.children)

    def get_consommation_global(self) -> float:
        return self.sum_leafs(self.tree)

    def get_consommation_by_parent(self, parent_name: str) -> float:
        node = tree_find_name(self.tree, parent_name)
        if node is None:
            raise ValueError(f"Parent {parent_name} not found")
        return self.sum_leafs(node)

    def get_basic_stats_leafs_consommation(self) -> float:
        consommations = np.array([leaf.consommation for leaf in self.tree.leaves])
        return np.mean(consommations), np.std(consommations), np.min(consommations), np.max(consommations)

    def construct_tree(self, limit_child_per_parent=None):
        result = []
        for parent_name, children in self.parents_enfants.items():
            result.extend(
                self.logement_name_to_node_name(parent_name, child)
                for child in children[:limit_child_per_parent]
            )

        self.path_dict = {k: {"consommation": round(random(), 1)} for k in result}  # todo
        self.tree = dict_to_tree(self.path_dict)

        # bring up consommation to parents
        # for node in self.tree.traverse("postorder"):
        print("The tree has been constructed.")

    def print(self, display_stats=True):
        if self.tree is None:
            raise ValueError("Tree is not constructed yet")

        if display_stats:
            n_leaves = len(list(self.tree.leaves))
            conso = self.get_consommation_global()
            print("-" * 1, "GLOBAL STATS", "-" * 35)
            print("* Consommation accumulés:", conso, "\n")
            print("-" * 1, f"LEAFS STATS ({n_leaves} logements)", "-" * 22)
            mean_, std_, min_, max_ = self.get_basic_stats_leafs_consommation()
            print("* Consommation moyenne:", mean_)
            print("* Consommation std:", round(std_))
            print("* Consommation min:", min_)
            print("* Consommation max:", max_)
            print("-" * 50)
        print_tree(self.tree, attr_list=["consommation"])
        print("-" * 50)


reseau = Reseau(limit_parents=15, limit_child_per_parent=2)

reseau.print()

print("done")

print("#")

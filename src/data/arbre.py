import re
from random import randint

import numpy as np
from bigtree import Node, print_tree, dict_to_tree

from datamanagers import get_dict_parents_enfants


# %%

def level_up_pl(pl_id: str) -> str:
    # keep digits
    id = re.sub(r"[^0-9]", "", pl_id)
    # print(id)
    depth = len(id)
    return 'PS' if depth <= 1 else pl_id[:-1]


# %%

class TreeManager():
    def __init__(self):
        self.tree = None
        self.construct_tree()

    def logement_name_to_node_name(self, parent_name, logement_name: str) -> str:
        global all_parents
        node_name = parent_name + "/" + logement_name
        grand_parent_name = parent_name
        for _ in range(4):
            grand_parent_name = level_up_pl(grand_parent_name)
            node_name = grand_parent_name + "/" + node_name
            if grand_parent_name == 'PS':
                break
        return node_name

    def add_conso_to_tree(self, logement_name: str, conso: float) -> None:
        pass  # todo

    def construct_tree(self):
        parents_enfants = get_dict_parents_enfants()
        result = []
        for parent_name, children in parents_enfants.items():
            result.extend(
                self.logement_name_to_node_name(parent_name, child)
                for child in children
            )
        path_dict = {k: {"consommation": randint(0, 1)} for k in result[:5]}  # todo

        self.tree = dict_to_tree(path_dict)

    def print(self):
        print("-" * 100)
        if self.tree is None:
            raise ValueError("Tree is not constructed yet")
        print_tree(self.tree, attr_list=["consommation"])

        n_leaves = len(list(self.tree.leaves))
        conso = self.get_consommation_accumulee()
        print("Nombre de logements:", n_leaves)
        print("-" * 40)
        print("Consommation accumulés:", conso)
        print("-" * 40)
        mean_, std_, min_, max_ = self.get_basic_stats_leafs_consommation()
        print("Consommation moyenne:", mean_)
        print("Consommation std:", std_)
        print("Consommation min:", min_)
        print("Consommation max:", max_)
        print("-" * 100)

    def __sum_leafs(self, node: Node) -> float:
        if node.is_leaf:
            return node.consommation
        else:
            return sum(self.__sum_leafs(child) for child in node.children)

    def get_consommation_accumulee(self) -> float:
        return self.__sum_leafs(self.tree)

    def get_basic_stats_leafs_consommation(self) -> float:
        consommations = np.array([leaf.consommation for leaf in self.tree.leaves])
        return np.mean(consommations), np.std(consommations), np.min(consommations), np.max(consommations)


tree_manager = TreeManager()
tree_manager.print()

print("done")

# %%



print("done")

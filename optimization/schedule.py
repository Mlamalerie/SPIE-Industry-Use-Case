import random
import re
from typing import List

import numpy as np
import pandas as pd
from bigtree import Node, print_tree, dict_to_tree
from bigtree import find_name as tree_find_name

from data.equipements import EquipementsDataManager
from data.limites_puissance import LimitesPuissanceDataManager
from data.relation_pl_maisons import get_dict_poste_livraisons_maisons

HEURES_CREUSES = ("20:00", "08:00")
HEURES_DEJEUNER = ("11:00", "13:00")
HEURES_DINER = ("18:00", "20:00")

MACHINES_HEURES_CREUSES = ["Md4-TV", "Mc1-FG", "Mc2-CE", "Mc3-CG", "Md5-FO", "Md6-PL", "Mc4-FG", "Mc5-CE"]


# %%


def hour_2_index(hour: str) -> int:
    """Convert hour to index in vector"""
    return int(hour.split(":")[0])


def put_value_randomly_on_vector(vector, value: float, how_many: int = 1, index_range: tuple = None,
                                 follow: bool = False):
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
        if not follow:
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
        if not follow:
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


def alterate_vector(vector, max_alteration: int = 1):
    """Put several times, randomly, the same value in different index of a vector
    :param vector: vector to modify
    """
    len_vector = len(vector)
    count = 0
    for i in range(len_vector):
        if vector[i] != 0:
            vector[i] = random.choice(vector)

            # random choose index on vector
            index = random.randint(0, len_vector - 1)
            # put value on vector
            vector[index] = vector[i]
            count += 1
            if count >= max_alteration:
                break

    return vector


"""z = np.zeros((3, 24), dtype=np.float32)
z[2] = put_value_randomly_on_vector(z[2], value=0.2, how_many=6, follow=True)


z[2] = alterate_vector(z[2], max_alteration=1)
"""


# %%
class Schedule:
    eqm = EquipementsDataManager()
    lpm = LimitesPuissanceDataManager()

    def __init__(self, logement_name, parent_name):
        self.logement_name = logement_name
        self.parent_name = parent_name
        self.logement_equipements: list = Schedule.eqm.get_equipements_by_logement_name(self.logement_name)
        self.genome: np.array = np.zeros((len(Schedule.eqm.equipements_names), 24), dtype=float)
        self.consommation: float = 0.0
        self.init_random_genome()

        # self.decrease_for_constraint_violation = 0.25  # 25% decrease in cost if constraint is violated

    def init_random_genome(self):
        # todo : tc and other
        # for each equipement in the logement
        # print("equipements_list", self.logement_equipements)
        for logement_equipement_name in self.logement_equipements:
            index_genome_matrix = Schedule.eqm.get_index_equipement_by_name(logement_equipement_name)
            # get caracteristics of equipement

            if caracteristiques := Schedule.eqm.caracteristiques_equipements.get(logement_equipement_name):
                # temps de cycle
                t_cycle = caracteristiques.get("tps_cycle")
                t_cycle = round(t_cycle, 0) if t_cycle > 1 else 1.0  # d'après jenna
                t_cycle = int(t_cycle) if logement_equipement_name != "Md4-TV" else random.randint(3, 5)

                # puissance par cycle
                puissance = caracteristiques.get("puissance")

                # sequencable : vaut 1 si aligné obligatoirement
                sequensable = bool(caracteristiques.get("sequensable"))

                # hr de debut
                hr_debut = caracteristiques.get(
                    "hr_debut")  # -1 si pas de contrainte, 1 si généré une fois dans heures creuses, N si générer N fois dans heures creuse

                how_many_to_put = int(t_cycle)
                value_to_put = puissance

                horaire_range = None
                if hr_debut > 0:
                    horaire_range = (hour_2_index(HEURES_CREUSES[0]), hour_2_index(HEURES_CREUSES[1]))
                    for _ in range(hr_debut):
                        self.genome[index_genome_matrix] = put_value_randomly_on_vector(
                            self.genome[index_genome_matrix],
                            value=value_to_put,
                            how_many=how_many_to_put,
                            index_range=horaire_range,
                            follow=not sequensable)
                else:
                    self.genome[index_genome_matrix] = put_value_randomly_on_vector(self.genome[index_genome_matrix],
                                                                                    value=value_to_put,
                                                                                    how_many=how_many_to_put,
                                                                                    index_range=horaire_range,
                                                                                    follow=sequensable)

                # if logement_equipement_name == "Mc2-CE":
                #    print("Mc2-CE", self.genome[index_genome_matrix])

            else:
                raise ValueError(f"caracteristiques not found for equipement")

        self.evaluate_consommation()

    def get_surface(self) -> int:
        return int(self.logement_name.split("-")[0][1:])  # todo delele all number

    def is_respect_limite_puissance_constraint(self, puissance_limite: float = None) -> bool:
        puissance_limite = Schedule.lpm.get_limites(self.get_surface())[
            "kVA"] if puissance_limite is None else puissance_limite
        sum_vect = np.sum(self.genome, axis=0)  # array avec sommes des puissance

        return np.all(sum_vect <= puissance_limite), np.sum(sum_vect > puissance_limite)

    def evaluate_consommation(self):
        self.consommation = np.sum(self.genome)

    def get_cost(self):
        cost = self.consommation
        ok, nb_violation = self.is_respect_limite_puissance_constraint()
        cost *= 1 if ok else nb_violation + 1
        return cost
        # todo : pénaliser si respect pas plage horaires creuses
        # todo : prendre en compte consommeation TV

    def get_consommation(self):
        return self.consommation

    def get_parent_name(self):
        return self.parent_name

    def get_logement_name(self):
        return self.logement_name

    def mutate(self):  # todo : add mutation
        # global MACHINES_HEURES_CREUSES
        # get indexes of genomes to mutate
        n = len(self.genome)
        indexes = np.random.choice(np.arange(n), size=random.randint(1, n - 1), replace=False)

        indexes_hc = [Schedule.eqm.get_index_equipement_by_name(equipement_hc) for equipement_hc in
                      MACHINES_HEURES_CREUSES]
        for i in indexes:
            if i in indexes_hc:
                self.genome[i] = alterate_vector(self.genome[i], max_alteration=1)
            else:
                self.genome[i] = alterate_vector(self.genome[i], max_alteration=random.randint(1, 2))

    def set_new_genome(self, genome):
        self.genome = genome
        self.evaluate_consommation()

    def to_df(self):
        hours_str = [f"{i:02d}:00" for i in range(24)]
        return pd.DataFrame(
            self.genome,
            columns=hours_str,
            index=Schedule.eqm.equipements_names,
        )


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
    def __init__(self, schedules: List[Schedule]):  # todo,

        self.tree = None
        self.global_cost: float = None
        self.base_leaves_paths_dict: dict = None
        self.leaves_parents_names = []

        self.construct_tree(schedules)
        self.load_global_cost()

    def logement_name_to_node_name(self, parent_name, logement_name: str) -> str:
        return "/".join(
            get_all_ancestors(parent_name) + [f"{parent_name}/{logement_name}"]
        )

    def sum_leaves(self, node: Node, what="cost") -> float:
        if node.is_leaf:
            return node.schedule.get_cost() if what == "cost" else node.schedule.get_consommation()
        else:
            return sum(self.sum_leaves(child) for child in node.children)

    def load_global_cost(self) -> None:
        self.global_cost = self.sum_leaves(self.tree, "cost")

    def get_global_cost(self) -> float:
        return self.global_cost

    def get_global_consommation(self) -> float:
        return self.sum_leaves(self.tree, "consommation")

    def __lt__(self, other):
        return self.global_cost < other.global_cost

    def get_cost_by_parent(self, parent_name: str) -> float:
        node = tree_find_name(self.tree, parent_name)
        if node is None:
            raise ValueError(f"Parent {parent_name} not found")
        return self.sum_leaves(node, "cost")

    def get_leaf_schedule_by_name(self, logement_name):
        leaf = tree_find_name(self.tree, logement_name)
        if leaf is None:
            raise ValueError(f"Leaf {logement_name} not found")
        return leaf.schedule

    def get_basic_stats_leaves_cost(self):
        costs = np.array([leaf.schedule.get_cost() for leaf in self.tree.leaves])
        return np.mean(costs), np.std(costs), np.min(costs), np.max(costs)

    def get_basic_stats_leaves_consommation(self):
        consommations = np.array([leaf.schedule.get_consommation() for leaf in self.tree.leaves])
        return np.mean(consommations), np.std(consommations), np.min(consommations), np.max(consommations)

    def construct_tree(self, schedules: List[Schedule], verbose=False):

        # self.path_dict = {k: {"cost": round(random(), 1)} for k in result}  # todo
        if not schedules:
            raise ValueError("No schedules provided")

        self.base_leaves_paths_dict = {}
        self.leaves_parents_names = []
        # for each schedule
        for schedule in schedules:
            self.leaves_parents_names.append(schedule.get_parent_name())
            leaf_path = self.logement_name_to_node_name(schedule.parent_name, schedule.logement_name)
            self.base_leaves_paths_dict[leaf_path] = {"schedule": schedule,
                                                      "consommation": round(schedule.consommation, 1)}

        self.tree = dict_to_tree(self.base_leaves_paths_dict)
        self.n_leaves = len(list(self.tree.leaves))

        if verbose:
            print("tree construction done.")

    def print(self, display_stats=True, display_tree=True, attr_list=None, f_print=print):
        if attr_list is None:
            attr_list = ["consommation"]
        if self.tree is None:
            raise ValueError("Tree is not constructed yet")

        f_print(f"id : {id(self)}")
        if display_stats and self.tree:
            self._print_stats(f_print=f_print)

        if display_tree:
            print_tree(self.tree, attr_list=attr_list, max_depth=5 if self.n_leaves > 100 else None)
        f_print("-" * 50)

    # TODO Rename this here and in `print`
    def _print_stats(self, f_print=print):
        n_leaves = len(list(self.tree.leaves))

        f_print("-" * 1, "GLOBAL STATS", "-" * 35)
        f_print("* Consommation global du réseau:", round(self.get_global_consommation(), 1), "KWh")
        f_print("* 'Cost':", round(self.get_global_cost(), 1))

        respects = [s.is_respect_limite_puissance_constraint()[0] for s in self.get_leaves_schedules()]
        f_print(f"* % Plannings avec les contraintes de puissances respectés: {np.sum(respects) / len(respects):.2%}",
                "\n")

        f_print("-" * 1, f"LEAVES STATS ({n_leaves} maisons)", "-" * 22)
        mean_consommation, std_consommation, min_consommation, max_consommation = self.get_basic_stats_leaves_consommation()
        f_print(
            f"* consommation: {mean_consommation:.2f} +- {std_consommation:.2f} (min: {min_consommation:.2f}, max: {max_consommation:.2f})")
        mean_cost, std_cost, min_cost, max_cost = self.get_basic_stats_leaves_cost()
        f_print(f"* 'cost': {mean_cost:.2f} +- {std_cost:.2f} (min: {min_cost:.2f}, max: {max_cost:.2f})")

    def get_schedule_by_leaf_path(self, leaf_path: str) -> Schedule:
        return self.base_leaves_paths_dict[leaf_path]["schedule"]

    def get_leaves_list(self) -> list:
        return list(self.tree.leaves)

    def get_leaves_schedules_by_node_parent(self, parent_name: str) -> List[Schedule]:
        node = tree_find_name(self.tree, parent_name)
        if node is None:
            raise ValueError(f"Parent {parent_name} not found")
        return [leaf.schedule for leaf in node.leaves]

    def get_leaves_schedules(self) -> List[Schedule]:
        return self.get_leaves_schedules_by_node_parent("PS")

    def _mutate_leaves(self, limit_leaves_mutations: int = None) -> list:
        if limit_leaves_mutations is None:
            limit_leaves_mutations = self.n_leaves
        limit_leaves_mutations = min(limit_leaves_mutations, self.n_leaves)
        n_mutations = random.randint(1, limit_leaves_mutations)
        # generate random indexes to mutate
        indexes = np.random.choice(self.n_leaves, n_mutations, replace=False)

        new_schedules = []
        for ix, leaf in enumerate(self.tree.leaves):
            # get schedule
            schedule = leaf.schedule
            # mutate
            if ix in indexes:
                schedule.mutate()
            new_schedules.append(schedule)

        self.construct_tree(new_schedules)
        return new_schedules

    def _mutate_leaves_and_sibling(self, n_leaves_to_focus: int, limit_leaves_mutations: int) -> None:
        if limit_leaves_mutations is None:
            limit_leaves_mutations = self.n_leaves

        limit_leaves_mutations = min(limit_leaves_mutations, self.n_leaves)

        pass
        for _ in range(n_leaves_to_focus):
            # get random leaf
            leaf = random.choice(self.get_leaves_list())
            # get siblings
            siblings = leaf.siblings
            # mutate all siblings
            for sibling in siblings:
                sibling.schedule.mutate()

        self.construct_tree([leaf.schedule for leaf in self.tree.leaves])

    def mutate(self, n_leaves_mutations=1, n_leaves_focus=1,
               rate: float = 0.7) -> None:  # todo n_leaves_mutations=N_MAISON_PAR_PL ?
        # Mutation de leaves : Cette méthode consiste à sélectionner au hasard un nœud de l'arbre et à le remplacer par un nœud aléatoire ou un nœud produit par une fonction de création de nœuds.

        if random.random() < rate:
            self._mutate_leaves(limit_leaves_mutations=n_leaves_mutations)
        else:
            self._mutate_leaves_and_sibling(n_leaves_to_focus=n_leaves_focus, limit_leaves_mutations=n_leaves_mutations)


def init_indivual(parents_enfants: dict) -> Reseau:
    schedules = [Schedule(logement, p) for p, logements in parents_enfants.items() for logement in logements]
    return Reseau(schedules=schedules)


if __name__ == "__main__":
    # reseau = Reseau()
    # reseau.print()
    rel_parents_enfants_maisons = get_dict_poste_livraisons_maisons(limite_poste_livraisons=15,
                                                                    limit_maisons_par_pl=100)
    example_schedules_load = [Schedule(logement, p) for p, logements in rel_parents_enfants_maisons.items() for logement
                              in logements]

    arezo = init_indivual(rel_parents_enfants_maisons)
    arezo.print()

    # example_schedules = [Schedule(logement, p) for p, logements in parents_enfants.items() for logement in logements]
    # reseau = Reseau(schedules=example_schedules)

    # reseau.print()

    print("done")

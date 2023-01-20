import os
from typing import List

import pandas as pd


# %%
def get_dict_poste_livraisons_maisons(limite_poste_livraisons: int = None, limit_maisons_par_pl: int = None,
                                      poste_livraisons: List[str] = None) -> dict:
    # provide absolute path to the data folder
    csv_file = f"{os.path.dirname(__file__)}/data lien reseau maison - parent et enfants.csv"
    df_parents_enfants = pd.read_csv(csv_file)
    df_parents_enfants_group = df_parents_enfants.groupby("Parent")["Enfant"].apply(list)

    if poste_livraisons is not None:
        # verify that all poste_livraisons exist in the data
        assert all([pl in df_parents_enfants_group.index for pl in
                    poste_livraisons]), "Some poste_livraisons are not in the data"

        df_parents_enfants_group = df_parents_enfants_group.loc[poste_livraisons]

    if limite_poste_livraisons is not None:
        df_parents_enfants_group = df_parents_enfants_group.head(limite_poste_livraisons)

    if limit_maisons_par_pl is not None:
        df_parents_enfants_group = df_parents_enfants_group.apply(lambda x: x[:limit_maisons_par_pl])
    return df_parents_enfants_group.to_dict()


# %%
"""
def get_dict_recap_type_logement() -> dict:
    csv_path: str = f"{os.path.dirname(__file__)}/data recap type logement.csv"
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="index")
"""

if __name__ == "__main__":
    rel_parents_enfants_maisons = get_dict_poste_livraisons_maisons(limit_maisons_par_pl=5,
                                                                    poste_livraisons=["PL1111", "PL11122"])
    print(rel_parents_enfants_maisons)

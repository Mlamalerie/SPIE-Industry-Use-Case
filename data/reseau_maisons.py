import os

import pandas as pd


# %%
def get_dict_parents_enfants(limit_parents: int = None, limit_child_per_parent: int = None) -> dict:
    # provide absolute path to the data folder
    csv_file = f"{os.path.dirname(__file__)}/data lien reseau maison - parent et enfants.csv"
    df_parents_enfants = pd.read_csv(csv_file)
    df_parents_enfants_group = df_parents_enfants.groupby("Parent")["Enfant"].apply(list)
    if limit_parents is not None:
        df_parents_enfants_group = df_parents_enfants_group.head(limit_parents)

    if limit_child_per_parent is not None:
        df_parents_enfants_group = df_parents_enfants_group.apply(lambda x: x[:limit_child_per_parent])
    return df_parents_enfants_group.to_dict()


# %%
"""
def get_dict_recap_type_logement() -> dict:
    csv_path: str = f"{os.path.dirname(__file__)}/data recap type logement.csv"
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="index")
"""

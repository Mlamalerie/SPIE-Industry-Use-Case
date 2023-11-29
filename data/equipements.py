import os

import numpy as np
import pandas as pd


class EquipementsDataManager:
    caracteristiques_csv_path: str = f"{os.path.dirname(__file__)}/data équipement maison - caracteristiques.csv"
    equipements_par_logements_csv_path: str = f"{os.path.dirname(__file__)}/data équipement maison - table.csv"

    def __init__(self, loading=True):
        # verify if caracteristiques_csv_path exists
        if not os.path.exists(self.caracteristiques_csv_path):
            raise FileNotFoundError(f"File {self.caracteristiques_csv_path} not found")
        # verify if equipements_par_logements_csv_path exists
        if not os.path.exists(self.equipements_par_logements_csv_path):
            raise FileNotFoundError(f"File {self.equipements_par_logements_csv_path} not found")

        self.equipements_list_par_logements: dict = None
        self.equipements_names = []
        self.caracteristiques_equipements: dict = {
            'Md1-LV': {'puissance': 1.3, 'tps_cycle': 1.0, 'amperage': 6.0, 'hr_debut': -1, 'hr_max': np.nan,
                       'sequensable': 0},
            'Md2-LL': {'puissance': 2.0, 'tps_cycle': 1.0, 'amperage': 9.0, 'hr_debut': -1, 'hr_max': np.nan,
                       'sequensable': 0},
            'Md3-SL': {'puissance': 1.0, 'tps_cycle': 4.0, 'amperage': 4.0, 'hr_debut': -1, 'hr_max': np.nan,
                       'sequensable': 0},
            'Md4-TV': {'puissance': 0.1, 'tps_cycle': np.nan, 'amperage': 0.5, 'hr_debut': 1, 'hr_max': np.nan,
                       'sequensable': 0},
            'Mc1-FG': {'puissance': 0.1, 'tps_cycle': 0.2, 'amperage': 0.5, 'hr_debut': 4, 'hr_max': np.nan,
                       'sequensable': 0},
            'Mc2-CE': {'puissance': 2.2, 'tps_cycle': 6.0, 'amperage': 10.0, 'hr_debut': 1, 'hr_max': np.nan,
                       'sequensable': 1},
            'Mc3-CG': {'puissance': 0.1, 'tps_cycle': 0.2, 'amperage': 0.5, 'hr_debut': 4, 'hr_max': np.nan,
                       'sequensable': 0},
            'Md5-FO': {'puissance': 1.6, 'tps_cycle': np.nan, 'amperage': 12.0, 'hr_debut': 1, 'hr_max': np.nan,
                       'sequensable': 0},
            'Md6-PL': {'puissance': 1.2, 'tps_cycle': np.nan, 'amperage': 20.0, 'hr_debut': 1, 'hr_max': np.nan,
                       'sequensable': 0},
            'Mc4-FG': {'puissance': 0.3, 'tps_cycle': 0.2, 'amperage': 3.0, 'hr_debut': 4, 'hr_max': np.nan,
                       'sequensable': 0},
            'Mc5-CE': {'puissance': 3.0, 'tps_cycle': 6.0, 'amperage': 15.0, 'hr_debut': 1, 'hr_max': np.nan,
                       'sequensable': 1}
        }
        self.equipements_names = list(self.caracteristiques_equipements.keys())
        if loading:
            # self.load_caracteristiques()
            self.load_equipements_list_par_logement()

    def get_index_equipement_by_name(self, equipement_name: str) -> int:
        return self.equipements_names.index(equipement_name)

    def load_caracteristiques(self) -> dict:
        df = pd.read_csv(self.caracteristiques_csv_path, header=0)
        df.set_index("Type", inplace=True)
        df = df.T

        # concat inex 'Type' and column 'Machine' to 'Type-Machine'
        df["Type-Machine"] = df.index + "-" + df["Machine"]
        df.set_index("Type-Machine", inplace=True, drop=True)
        df.drop(columns=["Machine"], inplace=True)
        # On last row, replace N values with 0, and O values with 1
        df["Sequensable"] = df["Sequensable"].replace({"N": 0, "O": 1})

        df["tps cycle"] = df["tps cycle"].astype(float)
        df["Puissance"] = df["Puissance"].astype(float)
        df["Amperage"] = df["Amperage"].astype(float)

        df["Hr debut"] = df["Hr debut"].replace(
            {"résultat de l'optimisation": -1, "à générer aléatoirement sur la plage HC": 1,
             "à générer 4 fois aléatoirement sur plage HC": 4})

        # lower column names, replace spaces with underscores
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        # invert index and columns and return a dict
        self.caracteristiques_equipements = df.to_dict(orient="index")


    def get_caracteristiques(self) -> dict:
        if self.caracteristiques_equipements is None:
            self.load_caracteristiques()
        return self.caracteristiques_equipements

    def get_df_equipements_par_logements(self, keep_logement_type_col=False) -> pd.DataFrame:
        df = pd.read_csv(self.equipements_par_logements_csv_path)
        df.columns = ["logement_name",
                      "logement_type"] + self.equipements_names  # /!\ depend du fichier caracteriques equioements
        if not keep_logement_type_col:
            df = df.drop(columns=["logement_type"])
        df.set_index("logement_name", inplace=True)
        return df

    def load_equipements_list_par_logement(self) -> None:
        """create a dict (key = logement_name, value = list of equipements who equals 1)"""
        df_equipements_par_logements = self.get_df_equipements_par_logements()
        equipements_list = {
            logement_name: equipements[equipements == 1].index.tolist()
            for logement_name, equipements in df_equipements_par_logements.iterrows()
        }
        self.equipements_list_par_logements = equipements_list

    def get_equipements_by_logement_name(self, logement_name) -> list:
        if self.equipements_list_par_logements is None:
            self.load_equipements_list_par_logement()
        equipements = self.equipements_list_par_logements.get(logement_name)
        if equipements is None:
            raise KeyError(f"Logement {logement_name} not found in equipements_list_par_logements")
        return equipements

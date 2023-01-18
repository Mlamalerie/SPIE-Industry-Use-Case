import os

import pandas as pd


class LimitesPuissanceDataManager():
    csv_path: str = f"{os.path.dirname(__file__)}/data équipement maison - limite de puissance par mc.csv"

    def __init__(self):
        self.limites_puissance: list = self.__get_limites_de_puissance_par_mc()

    def __get_limites_de_puissance_par_mc(self) -> pd.DataFrame:
        df = pd.read_csv(LimitesPuissanceDataManager.csv_path)
        return df.to_dict(orient="records")

    def get_limites(self, surface: float) -> dict:
        if surface < 0:
            raise ValueError("Surface must be positive")

        for limite in self.limites_puissance:
            if limite["Surface_inf"] <= surface < limite["Surface_sup"]:
                return limite

        return self.limites_puissance[-1]

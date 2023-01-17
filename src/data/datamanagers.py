import concurrent.futures
import os
from glob import glob
from typing import Dict
import matplotlib.pyplot as plt
import pandas as pd


# %%
class ConsommationDataManager:
    logements_loaded = {}  # todo : dont use {"df_conso"} and rename logements_dfs = {}

    conso_reseau_distribt_path = f"{os.path.dirname(__file__)}/conso_reseau_distriBT/"

    def __init__(self):
        self.logements_names = []
        self.load_logements_names()
        self.logements_types = []
        self.load_logements_types()

    def is_logement_loaded(self, logement_name: str) -> bool:
        return logement_name in self.logements_loaded

    def is_logement_name_exists(self, name: str) -> bool:
        return name in self.logements_names

    def is_logement_type_exists(self, logement_type: str) -> bool:
        return logement_type in self.logements_types

    def __display_loading_bar(self, i, n, bar_length=20):
        percent = float(i) / n
        arrow = '-' * int(round(percent * bar_length) - 1) + '>'
        spaces = ' ' * (bar_length - len(arrow))
        print("Loading: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))), end="\r")

    def get_logement_name_from_csv_file(self, csv_path: str) -> str:
        # get name (without ext) from csv path
        logement_name = os.path.splitext(os.path.basename(csv_path))[0]
        return logement_name.split("_")[-1]

    def load_logements_names(self) -> None:
        # get logement name from all csv files in all directories
        logements_names = []
        for dirpath in glob(f"{self.conso_reseau_distribt_path}/*/"):
            for csv_filepath in glob(dirpath + "*.csv"):
                logement_name = self.get_logement_name_from_csv_file(csv_filepath)
                logements_names.append(logement_name)

        self.logements_names = logements_names

    def load_logements_types(self) -> None:
        # get all directories in data_dir
        dirs = glob(self.conso_reseau_distribt_path + "*/")
        if len(dirs) == 0:
            raise FileNotFoundError(f"No directories found in {self.conso_reseau_distribt_path}")
        # get directory name for each directory path
        logements_types = [os.path.basename(os.path.normpath(dirpath)).split()[-1] for dirpath in dirs]
        self.logements_types = logements_types

    def load_df_conso_from_csv(self, csv_filepath: str) -> None:
        # read and parse 'date' column as datetime object (MM/DD/YYYY)
        # print(f"Loading {csv_filepath}")
        df = pd.read_csv(csv_filepath, header=1)

        # rename column "Unnamed: 1" to "consommation" and "Unnamed: 0" to "date"
        df.rename(columns={"Unnamed: 1": "consommation", "Unnamed: 0": "date"}, inplace=True)
        # parse date column
        df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y")
        # set index to "date" column
        df = df.set_index("date")

        logement_name = self.get_logement_name_from_csv_file(csv_filepath)
        self.logements_loaded[logement_name] = {"df_conso": df}

    def load_dfs_conso_from_logements_dirpath(self, dirpath: str) -> None:
        if not os.path.isdir(dirpath) or not os.path.exists(dirpath):
            raise FileNotFoundError(f"Directory {dirpath} not found")

        # get all csv files in dirpath (use glob)
        csv_files = glob(f"{dirpath}*.csv")
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No csv files found in {dirpath}")

        # use multithreading to load all csv files
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.load_df_conso_from_csv, csv_files)

    def load_dfs_conso_from_logement_type(self, logement_type: str) -> None:
        """
        Load all csv files from a logement type directory
        :param logement_type: logement type directory name (ex: A15-1, A25-1, A30-2)
        :return:
        """
        if not self.is_logement_type_exists(logement_type):
            raise ValueError(f"Logement type {logement_type} not exists")
        dirpath = f"{self.conso_reseau_distribt_path}/data {logement_type}/"

        self.load_dfs_conso_from_logements_dirpath(dirpath)

    def load_all_dfs_conso(self) -> None:
        data_dir = "conso_reseau_distriBT/"
        # get all directories in data_dir
        dirs = glob(f"{data_dir}*/")
        if len(dirs) == 0:
            raise FileNotFoundError(f"No directories found in {data_dir}")

        # load all directories
        print("Loading all...")
        for i, dirpath in enumerate(dirs):
            self.__display_loading_bar(i, len(dirs))
            self.load_dfs_conso_from_dir(dirpath)
        self.__display_loading_bar(i, len(dirs))

    def get_df_conso_by_logement_name(self, name: str) -> pd.DataFrame:
        if not self.is_logement_name_exists(name):
            raise ValueError(f"Logement {name} not exists")
        if not self.is_logement_loaded(name):
            if csv_query := glob(f"{self.conso_reseau_distribt_path}/*/data_maison_{name}.csv"):
                self.load_df_conso_from_csv(csv_query[0])
            else:
                raise KeyError(f"Logement {name} (data_maison_{name}.csv) not found")
        return self.logements_loaded[name]["df_conso"]

    def get_dfs_conso_by_logement_type(self, logement_type: str) -> Dict[str, pd.DataFrame]:
        if not self.is_logement_type_exists(logement_type):
            raise ValueError(f"Logement type {logement_type} not exists")

        return {
            logement_name: self.get_df_conso_by_logement_name(logement_name)
            for logement_name in self.logements_names
            if logement_name.startswith(logement_type)
        }


def get_dict_equipements_infos(csv_path: str = "data équipement maison - caracteristiques.csv", start_header=1) -> dict:
    # read csv file and swap the first two rows
    df = pd.read_csv(csv_path, header=start_header)
    df = df.set_index("Machine" if start_header == 1 else "Type")

    df = df.T
    # On last row, replace N values with 0, and O values with 1
    df["Sequensable"] = df["Sequensable"].replace({"N": 0, "O": 1})
    # invert index and columns and return a dict
    return df.to_dict(orient="index")


def get_dict_recap_type_logement(csv_path: str = "data recap type logement.csv") -> dict:
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="index")


def get_df_table_equipements_par_logements(csv_path: str = "data équipement maison - table.csv", keep_logement_type_col=False) -> pd.DataFrame:

    df = pd.read_csv(csv_path)
    equipements = get_dict_equipements_infos()
    df.columns = ["logement_name","logement_type"] + list(equipements.keys())
    if not keep_logement_type_col:
        df = df.drop(columns=["logement_type"])
    df.set_index("logement_name", inplace=True)
    return df


def get_limite_de_puissance_par_mc(
        csv_path: str = "data équipement maison - limite de puissance par mc.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    return df.to_dict(orient="records")



if __name__ == "__main__":
    cdm = ConsommationDataManager()
    print(cdm.get_df_conso_by_logement_name('A100-3-100'))
    equipements = get_dict_equipements_infos()
    #limites_puissance = get_limite_de_puissance_par_mc()
    table_equipements_par_logements = get_df_table_equipements_par_logements()
    recap_type_logement = get_dict_recap_type_logement()

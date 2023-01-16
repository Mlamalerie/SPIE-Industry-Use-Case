import concurrent.futures
import os
from glob import glob

import pandas as pd


# %%
class ConsommationDataManager:
    logements_loaded = {}
    conso_reseau_distribt_path = "conso_reseau_distriBT/"

    def __init__(self):
        self.logements_names = []
        self.load_logements_names()

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

    def load_df_conso_from_csv(self, csv_filepath: str) -> None:
        # read and parse 'date' column as datetime object (MM/DD/YYYY)
        # print(f"Loading {csv_filepath}")
        df = pd.read_csv(csv_filepath, parse_dates=['date'], index_col='date')
        # drop second row

        df = df.drop(df.index[1])
        logement_name = self.get_logement_name_from_csv_file(csv_filepath)
        self.logements_loaded[logement_name] = {"df_conso": df}

    def load_dfs_conso_from_logement_type(self, logement_type: str) -> None:
        """
        Load all csv files from a logement type directory
        :param logement_type: logement type directory name (ex: A15-1, A25-1, A30-2)
        :return:
        """
        dirpath = f"{self.conso_reseau_distribt_path}/data {logement_type}/"
        if not os.path.isdir(dirpath) or not os.path.exists(dirpath):
            raise FileNotFoundError(f"Directory {dirpath} not found")

        # get all csv files in dirpath (use glob)
        csv_files = glob(dirpath + "*.csv")
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No csv files found in {dirpath}")

        # use multithreading to load all csv files
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.load_df_conso_from_csv, csv_files)

    def load_all_dfs_conso(self) -> None:
        data_dir = "conso_reseau_distriBT/"
        # get all directories in data_dir
        dirs = glob(data_dir + "*/")
        if len(dirs) == 0:
            raise FileNotFoundError(f"No directories found in {data_dir}")
        # load all
        for i, dirpath in enumerate(dirs):
            self.__display_loading_bar(i, len(dirs))
            self.load_dfs_conso_from_dir(dirpath)
        self.__display_loading_bar(i, len(dirs))

    def get_df_conso_by_logement_name(self, name: str) -> pd.DataFrame:
        if name not in self.logements_names:
            raise ValueError(f"Logement name {name} not exists")
        if name not in self.logements_loaded:
            if csv_query := glob(f"{self.conso_reseau_distribt_path}/*/data_maison_{name}.csv"):
                self.load_df_conso_from_csv(csv_query[0])
            else:
                raise KeyError(f"Logement {name} not found")
        return self.logements_loaded[name]["df_conso"]


dm = ConsommationDataManager()
# dm.load_dfs_conso_from_logement_type("A25-1")
dm.get_df_conso_by_logement_name("A25-1-200")

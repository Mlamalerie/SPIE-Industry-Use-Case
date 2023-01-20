import pandas as pd

from data.consommation import ConsommationDataManager
from prediction.prediction import *

consommation_manager = ConsommationDataManager()

# dict = get_dict_parents_enfants()

# for key, value in dict.items():
#     for logement in value:
#         if not consommation_manager.is_logement_name_exists(logement):
#             print("logement " + logement)
#         else:
#             ts = time_series(logement, consommation_manager=consommation_manager)
#             total_prediction_hw(ts)
# consommation_manager.get_logement_name_from_csv_file

def get_prediction_general():
    recap = pd.read_csv("data/data recap type logement.csv")
    recap["consomation moyenne"] = 0.0
    recap["consomation total"] = 0.0
    sample = 2

    total = 0.0
    for index, row in recap.iterrows():
        total_type = 0.0
        for i in range(sample):
            logement = row["type logement"] + "-" + str(i + 1)
            ts = time_series(logement, consommation_manager=consommation_manager)
            total_type += total_prediction_hw(ts)[1079] / sample
        recap.at[index, "consomation moyenne"] = 4.0
        total_type = total_type * row["nb logement"]
        recap.at[index, "consomation total"] = total_type

        total += total_type

    recap["consomation percentage"] = recap["consomation total"] / total

    return recap

prediction_general = get_prediction_general()
print(prediction_general)
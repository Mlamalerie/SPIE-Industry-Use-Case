from typing import Dict
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from data.consommation import ConsommationDataManager

def time_series(logement_name: str, consommation_manager: ConsommationDataManager = ConsommationDataManager()) -> Dict[str, pd.Series]:

    df_logement = consommation_manager.get_df_conso_by_logement_name(logement_name)
    series_logement = {}
    for column in df_logement.columns:
        series_logement[column] = df_logement[column]
    return series_logement

def solo_prediction_hw(time_serie: pd.Series) -> pd.Series:

    hw = ExponentialSmoothing(time_serie, trend = 'add', damped_trend = True, seasonal = 'mul', seasonal_periods = 365).fit()
    return hw.forecast()

def total_prediction_hw(time_series: Dict[str, pd.Series]) -> pd.Series:

    return solo_prediction_hw(time_series['consommation'])

def time_predictions_hw(time_series: Dict[str, pd.Series]) -> Dict[str, pd.Series]:

    predictions = {}
    for key in time_series:
        if key != 'consommation':
            predictions[key] = solo_prediction_hw(time_series[key])
    return predictions
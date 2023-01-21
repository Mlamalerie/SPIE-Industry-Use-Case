import matplotlib.pyplot as plt
import streamlit as st

from prediction.prediction import *

st.title("Prediction")

st.write('Predictions effectuées sur un logement à J+1 en fonction des consommations relevées par le passé')

options = {'Type': ['Appartement', 'Maison'],
           'Surface': [15, 25, 30, 50, 65, 80, 85, 100, 110, 120, 130, 135, 140, 150, 160, 200],
           'Habitants': [1, 2, 3, 4, 5, 6]}

option_type = st.selectbox('Quel est votre type de logement ?', options['Type'])

option_surface = st.selectbox('Quelle est la surface de votre logement (en m²) ?', options['Surface'])

option_habitants = st.selectbox("Combien d'habitants résident dans votre logement ?", options['Habitants'])

number = st.number_input("Insérez l'id de votre logement", min_value=1, value=2)

logement = '{}{}-{}-{}'.format(option_type[0], option_surface, option_habitants, number)
series = time_series(logement)

option_consommation_totale = st.checkbox('Consommation totale')

if option_consommation_totale:
    consommation_totale = total_prediction_hw(series)
    st.subheader('Demain, vous consommerez un total de {:.1f} kWh.'.format(consommation_totale))

option_consommation_horaire = st.checkbox('Consommation horaire')

if option_consommation_horaire:
    consommation_horaire = time_predictions_hw(series)
    st.subheader('Voici les détails de votre consommation prévue pour demain :')
    etiquettes, valeurs = zip(*consommation_horaire)
    fig, ax = plt.subplots()
    ax.bar(etiquettes, valeurs)
    st.pyplot(fig)
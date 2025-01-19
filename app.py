import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Charger le modèle LSTM
model = load_model('lstm_model.h5')

# Titre de l'application
st.title("Prédiction des prix de l'or avec LSTM")

# Charger les données
df = pd.read_csv('gold_prices.csv')

# Afficher les premières lignes
st.write(df.head())

# Normalisation des données
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Faire les prédictions
X_test = ...  # Charger les données de test
predictions = model.predict(X_test)

# Inverser la normalisation
predictions = scaler.inverse_transform(predictions)

# Visualiser les prédictions
st.write("Visualisation des prédictions")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(predictions, label='Prédictions')
ax.plot(df['Close'], label='Données réelles')
st.pyplot(fig)

# Fonctionnalités interactives
st.slider("Sélectionner la période", min_value=1, max_value=100, step=1)

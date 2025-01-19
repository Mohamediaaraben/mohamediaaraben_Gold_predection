# mohamediaaraben_Gold_predection

Projet de prédiction des prix de l'or et crypto
Rapport sur le Projet de Prédiction des Prix de l'Or avec LSTM
Titre : Prédiction des Prix de l'Or à l'aide de LSTM (Long Short-Term Memory)
Auteur : Mohamed Iaaraben
Date : Janvier 2025
1. Introduction
Le prix de l'or est un indicateur clé sur les marchés financiers et joue un rôle crucial dans la prise de décisions économiques et financières. La prédiction du prix de l'or est donc un sujet très recherché. Ce projet a pour objectif de prédire les prix futurs de l'or en utilisant un modèle d'apprentissage profond basé sur les réseaux de neurones LSTM (Long Short-Term Memory), particulièrement adapté aux séries temporelles. Le modèle sera formé à partir de données historiques des prix de l'or, récupérées à l'aide de yfinance.

2. Objectifs du Projet
L'objectif principal de ce projet est de :

Utiliser les données historiques des prix de l'or pour entraîner un modèle LSTM.
Développer un tableau de bord interactif avec Streamlit pour visualiser les prédictions.
Générer des prédictions sur les prix futurs de l'or à partir du modèle formé.
Fournir un outil pratique pour explorer les prix passés et futurs de l'or.
3. Méthodologie
3.1. Récupération des Données
Les données historiques des prix de l'or sont obtenues via la bibliothèque yfinance. Le symbole de l'or est 'GC=F' sur Yahoo Finance, et les données sont téléchargées sur une période allant de 2010 à 2025. Nous utilisons le prix de clôture de chaque jour comme principale donnée pour prédire les prix futurs.

3.2. Prétraitement des Données
Les données récupérées sont ensuite prétraitées pour :

Convertir la colonne Date en format datetime.
Normaliser les prix de l'or à l'aide du MinMaxScaler pour les ajuster entre 0 et 1.
Créer des ensembles de données d'entraînement et de test à l'aide d'une fenêtre de temps (look-back) de 60 jours.
3.3. Entraînement du Modèle LSTM
Un modèle LSTM est formé pour prédire les prix futurs en utilisant les données d'entraînement. Le modèle comprend :

Deux couches LSTM pour extraire les tendances à court et long terme.
Des couches Dropout pour éviter le sur-apprentissage.
Une couche Dense pour la prédiction finale.
3.4. Prédictions et Visualisation
Une fois le modèle formé, des prédictions sont générées pour les prix futurs de l'or. Ces prédictions sont visualisées sur un graphique avec les données réelles et les valeurs prédites. Un tableau de bord interactif est créé avec Streamlit pour afficher les résultats.

4. Résultats
4.1. Données Historiques de l'Or
Les premières lignes du fichier des données historiques de l'or sont comme suit :

csv
Copier
Modifier
Date,Open,High,Low,Close,Adj Close,Volume
2010-01-04,1097.80,1100.90,1096.30,1098.80,1098.80,49330
2010-01-05,1098.10,1100.60,1094.50,1098.10,1098.10,50600
...
4.2. Visualisation des Prédictions
Un graphique a été généré pour comparer les valeurs réelles des prix de l'or et les valeurs prédites par le modèle LSTM. Ce graphique montre l'exactitude du modèle en prédisant les tendances des prix.

5. Code Source
5.1. Téléchargement des Données avec yfinance
python
Copier
Modifier
import yfinance as yf

# Télécharger les données des prix de l'or
gold = yf.download('GC=F', start='2010-01-01', end='2025-01-01')

# Sauvegarder les données dans un fichier CSV
gold.to_csv('gold_prices.csv')
5.2. Prétraitement des Données
python
Copier
Modifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Charger les données
df = pd.read_csv('gold_prices.csv')

# Convertir la colonne de date
df['Date'] = pd.to_datetime(df['Date'])

# Normaliser les prix de l'or
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Créer les ensembles de données d'entraînement
look_back = 60
X, y = [], []
for i in range(look_back, len(df)):
    X.append(df['Close'].values[i - look_back:i])
    y.append(df['Close'].values[i])

X = np.array(X)
y = np.array(y)
5.3. Entraînement du Modèle LSTM
python
Copier
Modifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X, y, epochs=10, batch_size=32)

# Sauvegarder le modèle
model.save('lstm_model.h5')
5.4. Visualisation avec Streamlit
python
Copier
Modifier
import streamlit as st
import matplotlib.pyplot as plt

# Charger le modèle LSTM
model = load_model('lstm_model.h5')

# Afficher les données historiques et les prédictions
st.title("Prédiction des prix de l'or")
st.write(df.head())

# Visualiser les résultats
fig, ax = plt.subplots()
ax.plot(predictions, label='Prédictions')
ax.plot(df['Close'], label='Données réelles')
st.pyplot(fig)
6. Conclusion
Le modèle LSTM a été utilisé avec succès pour prédire les prix de l'or en se basant sur les données historiques. Ce projet a permis de démontrer la capacité des réseaux neuronaux à modéliser et à prédire les séries temporelles, en particulier pour des actifs financiers comme l'or. Le tableau de bord Streamlit permet une visualisation dynamique et interactive des prédictions, ce qui peut être très utile pour les analystes financiers et les traders.

7. Références
yfinance : https://pypi.org/project/yfinance/
TensorFlow : https://www.tensorflow.org/
Streamlit : https://streamlit.io/
8. Annexes
Les fichiers associés au projet incluent :

gold_prices.csv : Les données historiques des prix de l'or.
lstm_model.h5 : Le modèle LSTM entraîné.
predictions_gold_prices.xlsx : Les prédictions sauvegardées.
Lien pour accéder aux fichiers du projet
Tu peux trouver l'ensemble des fichiers nécessaires à ce projet sur mon dépôt GitHub à l'adresse suivante :
Projet de prédiction des prix de l'or - GitHub

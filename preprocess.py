
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Charger les données
df = pd.read_csv('gold_prices.csv', date_parser=True)

# Convertir la colonne de date au format datetime
df['Date'] = pd.to_datetime(df['Date'])

# Utiliser la colonne "Close" pour la prédiction
df = df[['Date', 'Close']]

# Normalisation des prix de l'or
scaler = MinMaxScaler(feature_range=(0, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Créer les ensembles de données pour l'entraînement et le test
X = []
y = []
look_back = 60

for i in range(look_back, len(df)):
    X.append(df['Close'].values[i - look_back:i])
    y.append(df['Close'].values[i])

X = np.array(X)
y = np.array(y)

# Sauvegarder les ensembles d'entraînement et de test dans des fichiers CSV
pd.DataFrame(X).to_csv('X_data.csv', index=False)
pd.DataFrame(y).to_csv('y_data.csv', index=False)

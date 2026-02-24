import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor

# Création du dossier
os.makedirs("models/models", exist_ok=True)

# Chargement  les données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Charger des meilleurs paramètres sauvegardés dans le pkl 
with open("models/models/best_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Création du modèle avec les meilleurs paramètres
model = RandomForestRegressor(**best_params, random_state=42)

# Entraîner et sauvegarde
model.fit(X_train, y_train)

with open("models/models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("fin de l'entrainement.")

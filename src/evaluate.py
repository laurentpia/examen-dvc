import os
import json
import pickle
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

# Création du dossier si non créé
os.makedirs("metrics", exist_ok=True)

# Chargement du modèle entrainé
with open("models/models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Chargement du modèle
X_test = pd.read_csv("data/processed/X_test_scaled.csv")
y_test = pd.read_csv("data/processed/y_test.csv")["silica_concentrate"]

# Prédictions
y_pred = model.predict(X_test)

# Sauvegarde des prédictions
predictions = pd.DataFrame({
    "y_true": y_test,
    "y_pred": y_pred
})
predictions.to_csv("data/predictions.csv", index=False)

# Calcul et affichage des métriques
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

scores = {
    "mse": mse,
    "r2": r2
}

# Sauvegarde des scores
with open("metrics/scores.json", "w") as f:
    json.dump(scores, f)

print("Évaluation terminée.")
print(scores)

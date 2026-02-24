import os
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Créer dossier models si nécessaire
os.makedirs("models/models", exist_ok=True)

# Charger les données
X_train = pd.read_csv("data/processed/X_train_scaled.csv")
y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

# Définir le modèle
model = RandomForestRegressor(random_state=42)

# Grille de paramètres
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}

# GridSearch
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

# Sauvegarde des meilleurs paramètres
with open("models/models/best_params.pkl", "wb") as f:
    pickle.dump(grid.best_params_, f)

print("GridSearch terminé.")
print("Meilleurs paramètres :", grid.best_params_)

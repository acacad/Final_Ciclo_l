import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn import tree

# Charger le dataset
columns = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
df = pd.read_csv('abalone.data', names=columns)

# Mélanger les données
df = shuffle(df, random_state=0)

# Encoder la variable catégorique "Sex"
label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])  # F = 0, I = 1, M = 2

# Séparer les variables explicatives (features) et la variable cible (target)
X = df.drop('Rings', axis=1)  # Toutes les colonnes sauf "Rings"
y = df['Rings']  # La colonne "Rings" est la cible

# Diviser les données en ensemble d'entraînement + validation (80%) et de test (20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Diviser les données d'entraînement + validation en ensemble d'entraînement (70%) et de validation (30%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.3, random_state=42)

# Meilleurs paramètres fournis
best_params = {
    'criterion': 'squared_error',
    'splitter': 'best',
    'max_depth': 6,
    'min_samples_split': 2,
    'min_samples_leaf': 5,
    'max_features': None,
    'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0,
    'ccp_alpha': 0.1  # Added ccp_alpha for pruning
}
parameters_to_test = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 6, 7, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2'],
    'max_leaf_nodes': [None, 10, 20, 50],
    'min_impurity_decrease': [0.0, 0.01, 0.1]
}

# Créer un modèle d'arbre de décision pour la régression avec les meilleurs paramètres
regressor = DecisionTreeRegressor(**best_params, random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
regressor.fit(X_train, y_train)

# Faire des prédictions sur l'ensemble de validation
y_val_pred = regressor.predict(X_val)

# Calculer le Mean Squared Error et le R-squared pour l'ensemble de validation
val_mse = mean_squared_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Validation Mean Squared Error: {val_mse:.4f}")
print(f"Validation R-squared: {val_r2:.4f}")

# Faire des prédictions sur l'ensemble de test
y_test_pred = regressor.predict(X_test)

# Calculer le Mean Squared Error et le R-squared pour l'ensemble de test
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Afficher les résultats pour l'ensemble de test
print(f"Test Mean Squared Error: {test_mse:.4f}")
print(f"Test R-squared: {test_r2:.4f}")

# Visualiser l'arbre de décision (optionnel)
plt.figure(figsize=(20, 20))  # Ajuster la taille de la figure si nécessaire
tree.plot_tree(regressor, feature_names=X.columns, filled=True)
plt.title('Arbre de décision pour prédire les Rings')
plt.show()

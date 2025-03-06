# QDA-avec-iris-dataset
programme python

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix

# Chargement du dataset Iris
iris = datasets.load_iris()
X = iris.data  # Caractéristiques (features)
y = iris.target  # Labels (0, 1, 2 pour les 3 classes)

# Séparation des données en train et test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Entraînement du modèle QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train, y_train)

# Prédiction sur le jeu de test
y_pred = qda.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle QDA : {accuracy:.2f}")

# Matrice de confusion
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, cmap="Blues", fmt="d", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.title("Matrice de confusion - QDA sur Iris")
plt.show()

Application de Détection de Fraude par Carte de Crédit

Description du Projet

Cette application détecte les fraudes dans les transactions par carte de crédit à l'aide de modèles d'apprentissage automatique. Elle est construite avec Streamlit pour l'interface utilisateur et utilise scikit-learn pour l'entraînement des modèles.

Fonctionnalités :
Chargement des données : Visualisez un échantillon des données de transactions à partir du fichier creditcard.csv.
Sélection du modèle : Choisissez entre trois modèles d'apprentissage automatique :
Random Forest
SVM (Support Vector Machine)
Régression Logistique
Personnalisation des hyperparamètres : Ajustez les paramètres du modèle, comme le nombre d'arbres pour Random Forest.
Métriques de performance : Affiche l'accuracy, la précision et le rappel du modèle.
Graphiques de performance : Matrice de confusion, courbe ROC, et courbe Précision-Rappel.
Prérequis
Python 3.x installé
Bibliothèques Python nécessaires :
streamlit
pandas
numpy
scikit-learn
matplotlib

Installation
Cloner le projet :git clone https://github.com/laetyyyy/Credit_Card_Fraud_Detection.git

cd Credit_Card_Fraud_Detection

Créer un environnement virtuel (optionnel mais recommandé) :
python -m venv .venv
source .venv/bin/activate  # Sur Linux/Mac
.venv\Scripts\activate  # Sur Windows

Installer les dépendances :
pip install -r requirements.txt

Exécution de l'Application
Lancer l'application :streamlit run app.py
Accéder à l'application : Ouvrez votre navigateur et allez à l'adresse suivante :http://localhost:8501

Utilisation de l'application :
Chargez et visualisez un échantillon des données de transactions.
Sélectionnez un modèle d'apprentissage automatique (Random Forest, SVC ou Régression Logistique).
Ajustez les hyperparamètres du modèle (si nécessaire).
Cliquez sur "Exécuter" pour entraîner le modèle et afficher les résultats.
Métriques de Performance
Accuracy : Pourcentage de prédictions correctes.
Précision : Proportion de transactions frauduleuses correctement identifiées.
Rappel : Proportion des cas de fraude réellement détectés par le modèle.

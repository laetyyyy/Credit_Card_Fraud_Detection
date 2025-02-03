import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt

def main():
    st.title("Application pour la détection de fraude par carte de crédit")
    st.subheader("Auteur : LEMOU Laeticia")

    # Train/test Split 
    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv('creditcard.csv')
        return data

    # Affichage de la base de données
    df = load_data()
    df_sample = df.sample(100)
    if st.sidebar.checkbox("Afficher les données brutes", False):
        st.subheader("Jeu de données 'creditcard': Échantillons de 100 observations")
        st.write(df_sample)

    seed = 123 

    # Fonction de prétraitement
    @st.cache_data(persist=True)
    def split(df):
        y = df['Class']
        X = df.drop('Class', axis=1)
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.2,
            stratify=y,
            random_state=seed
        )
        return x_train, x_test, y_train, y_test  # Ajout du return

    x_train, x_test, y_train, y_test = split(df)  # Correction ici
    classifier = st.sidebar.selectbox(
        "Choisissez un classifieur",
        ["SVC", "Random Forest", "Logistic Regression"]
    )

    if classifier == "Random Forest":
        st.sidebar.subheader("Hyperparametres du modeles ")
        n_arbres = st.sidebar.number_input(
            "choisir le Nombre d'arbres dans la foret",
            100 ,1000,step=10
        )
        profondeur_arbre = st.sidebar.number_input(
            "choisir la profondeur maximale d'un arbres",
            1, 20, step=1
        )
        bootstrap_arbre = st.sidebar.radio(
            "Echantillons bootstrap lors de la creation d'arbres ?",
            ["True", "False"]

        )    
        if st.sidebar.button("Execution",key ="Classify"):
            st.subheader("Random Forest Resultats")

            #Initialisation d'un objet RandomForestClassifier
            model = RandomForestClassifier(
            n_estimators  = n_arbres,
            max_depth=  profondeur_arbre,
            bootstrap = bootstrap_arbre 
            )
            #Entrainement de l'algorithme
            model.fit(x_train, y_train)

            #predrictions
            y_pred = model.predict(x_test)

            #Metriques de la performance 
            accuracy = model.score(x_test,y_test)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')

            #Afficher les metriques dans l'application 
            st.write("Accuarcy ",accuracy.round(3))
            st.write("Precision ",precision.round(3))
            st.write("Recall ",recall.round(3))




if __name__ == '__main__':
    main()

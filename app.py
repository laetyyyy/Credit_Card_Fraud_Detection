import streamlit as st 
import pandas as pd 
import numpy as np 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score
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

    #Analyse de la performance des modeles 
    
    
    def plot_perf(graphes,x_test, y_test, y_pred, model):
        if 'Confusion  matrix' in graphes:
            st.subheader('Matrice de confusion') 
            fig, ax = plt.subplots()  # Create a new figure and axis
            ConfusionMatrixDisplay.from_estimator(
                model,
                x_test,
                y_test,
                ax=ax  # Plot on the created axis
            )
            st.pyplot(fig)  # Pass the figure to st.pyplot()

        if 'ROC curve' in graphes:
            st.subheader('Courbe ROC') 
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            fig, ax = plt.subplots()  # Create a new figure and axis
            RocCurveDisplay(fpr=fpr, tpr=tpr).plot(ax=ax)  # Plot on the created axis
            st.pyplot(fig)  # Pass the figure to st.pyplot()

        if 'Precision-Recall curve' in graphes:
            st.subheader('Courbe Precision-Recall') 
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            fig, ax = plt.subplots()  # Create a new figure and axis
            PrecisionRecallDisplay(precision=precision, recall=recall).plot(ax=ax)  # Plot on the created axis
            st.pyplot(fig)  # Pass the figure to st.pyplot()

        

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
            [True, False]
        )   

        graphes_perf = st.sidebar.multiselect(
            "Choisir les graphes de performances",
            ("Confusion  matrix","ROC curve","Precison-Recall curve")

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
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)

            #Afficher les metriques dans l'application 
            st.write("Accuracy ", round(accuracy, 3))
            st.write("Precision ",round(precision, 3))
            st.write("Recall ",round(recall, 3))

            #Afficher les graphes de performances dans l'application
            plot_perf(graphes_perf)


if __name__ == '__main__':
    main()

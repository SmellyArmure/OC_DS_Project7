import streamlit as st
import numpy as np
import pandas as pd
import dill
import requests


def request_prediction(model_uri, data):

    headers = {"Content-Type": "application/json"}
    data_json = {'data': data}
    response = requests.request(method='POST',
    							headers=headers,
    							url=model_uri,
    							json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code,
            										   response.text))

    return response.json()

# # OPTION 1 : Chargement des données
# with open(path_file, 'rb') as file: 
# 	dict_cleaned = dill.load(file)
# OPTION 2 : Mise en cache de la fonction pour exécution unique
@st.cache
def load_data(path):
    # import the data in the .py file
	with open(path, 'rb') as f:
		dict_cleaned = dill.load(f)
	return dict_cleaned

# import best model and best threshold

f = open(r"thresh.txt", "r")
thresh_retrieved = f.read()
f.close()


# import the data in the .py file
path_file = '..\..\PICKLES\dict_cleaned.pkl'
dict_data = load_data(path_file)
X_train = dict_data['X_train']
y_train = dict_data['y_train']
X_test = dict_data['X_test']

id_train = X_train.index.tolist()
id_test = X_test.index.tolist()

##############################################

st.text('Maryse MULLER - Parcours Data Science projet 7 - OpenClassrooms')

st.title('Dashboard "Prêt à dépenser"')
st.subheader("Customer scoring")

# # for simple text input
# id_input = st.text_input('Client ID: ', id_test)

# for drop down menu (combo box)
id_input = st.selectbox("Please choose from the customer's list", id_test)

st.write("L'ID saisi est " + str(id_input))

st.markdown(chaine_features)
st.subheader('Modifier le profil client')
st.sidebar.header("Modifier le profil client")
st.sidebar.markdown('Cette section permet de modifier une ')
st.write(df_explanation, unsafe_allow_html=True)


def main():

    MLFLOW_URI = 'http://127.0.0.1:5000/invocations'

    predict_btn = st.button('Prédire')
    # Si appui sur le bouton prédire
    if predict_btn:
    	# valeurs de X d'un client
    	data = X_test.loc[id_input].values
        # la dashboard demande la prédiction au modèle préalablement déployé en local
		y_pred = request_prediction(MLFLOW_URI, data_X)[0]

        st.write(
            'Score du client: {:.2f}'.format(pred_y))

if __name__ == '__main__':
    main()
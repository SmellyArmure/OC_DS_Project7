# TO RUN : $streamlit run dashboard/dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501
# Online URL : http://15.188.179.79

import streamlit as st
from PIL import Image
import requests
import json
import pandas as pd
import matplotlib.pyplot as plt


def main():
    # local API
    API_URL = "http://127.0.0.1:5000/api/"

    ##################################
    #### LIST OF API REQUEST FUNCTIONS

    # Get list of SK_IDS (cached)
    @st.cache
    def get_sk_id_list():
        # URL of the sk_id API
        SK_IDS_API_URL = API_URL + "sk_ids/"
        # Requesting the API and saving the response
        response = requests.get(SK_IDS_API_URL)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of SK_IDS from the content
        SK_IDS = content['data']
        return SK_IDS

    # Get Personal data (cached)
    @st.cache
    def get_data_cust(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        PERSONAL_DATA_API_URL = API_URL + "data_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response to API request
        response = requests.get(PERSONAL_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        data_cust = pd.Series(content['data']).rename("SK_ID {}".format(select_sk_id))
        return data_cust

    # Get data from 20 nearest neighbors in train set (cached)
    @st.cache
    def get_data_neigh(select_sk_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        NEIGH_DATA_API_URL = API_URL + "neigh_cust/?SK_ID_CURR=" + str(select_sk_id)
        # save the response of API request
        response = requests.get(NEIGH_DATA_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.Series
        neigh_cust = pd.DataFrame(content['data'])
        return neigh_cust

    # Get scoring (cached)
    @st.cache
    def get_cust_scoring(select_sk_id):
        # URL of the scoring API
        SCORING_API_URL = API_URL + "scoring_cust/?SK_ID_CURR=" + str(select_sk_id)
        # Requesting the API and save the response
        response = requests.get(SCORING_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # getting the values from the content
        score = content['score']
        return score

    # Get the list of features
    @st.cache
    def get_features_descriptions():
        # URL of the aggregations API
        FEAT_DESC_API_URL = API_URL + "feat_desc"
        # Requesting the API and save the response
        response = requests.get(FEAT_DESC_API_URL)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        features_desc = pd.Series(content['data']['Description']).rename("Description")
        return features_desc

    #################################
    #################################
    #################################

    # Display the logo
    image = Image.open('dashboard/logo.png')
    st.sidebar.image(image, width=180)
    # Display the title
    st.title('Dashboard - Loan application customer scoring"')

    ##################################################
    # Select the customer's ID
    #################################################

    SK_IDS = get_sk_id_list()
    select_sk_id = st.sidebar.selectbox('Select SK_ID from list:', SK_IDS, key=1)
    st.write('You selected: ', select_sk_id)

    ##################################################
    # FEATURES' IMPORTANCE (SHAP VALUES) for 20 nearest neighbors
    ###############################################

    # st.header('SHAP Features importance\n(20 nearest neighbors)')
    # st.sidebar('SHAP Features importance\n(20 nearest neighbors)')

    # # choice display button
    # if st.sidebar.checkbox('Show global interpretation'):

    # get shap's values
    # shap_values = get_shap_values()
    # # draw the graph
    # fig, ax = plt.subplots()
    # ax.pie(frequencies)

    # # Plot the graph on th dashboard
    # st.pyplot()

    # if st.checkbox('Show details'):
    #     st.dataframe(shap_values)


    ##################################################
    # PERSONAL DATA
    ##################################################
    st.header('PERSONAL DATA BOXPLOT')

    if st.sidebar.checkbox('Show personal data'):
        # Get personal data
        cust_data = get_data_cust(select_sk_id)

        if st.checkbox('Show comparison with 20 neighbors data'):
            # Get 20 neighbors personal data
            neigh_data = get_data_neigh(select_sk_id)
            # aggregate the values
            neigh_data_agg = pd.DataFrame(neigh_data).mean().rename('Mean on 5000 sample')
            # Concatenation of the information to display
            df_display = pd.concat([cust_data, neigh_data_agg], axis=1)
        else:
            # Display only personal_data
            df_display = cust_data
        st.dataframe(df_display)

    ##################################################
    # SCORING
    ##################################################
    st.header('DEFAULT PROBABILITY')

    if st.sidebar.checkbox('Show default probability'):
        # Get score
        score = get_cust_scoring(select_sk_id)
        # Display score (default probability)
        st.write('Default probability:', score)

        if st.checkbox('Show explanations'):
            a=1
            # # Get prediction, bias and features contribs from surrogate model
            # (_, bias, contribs) = score_explanation(select_sk_id)
            # # Display the bias of the surrogate model
            # st.write("Population mean (bias):", bias)
            # # Remove the features with no contribution
            # contribs = contribs[contribs!=0]
            # # Sorting by descending absolute values
            # contribs = contribs.reindex(contribs.abs().sort_values(ascending=False).index)

            # st.dataframe(contribs)

    ##################################################
    # FEATURES DESCRIPTIONS
    ##################################################
    st.header("FEATURES' DESCRIPTIONS")
    
    features_desc = get_features_descriptions()

    if st.sidebar.checkbox('Show features descriptions'):
        # Display features' descriptions
        st.table(features_desc)
    
    ################################################


if __name__== '__main__':
    main()
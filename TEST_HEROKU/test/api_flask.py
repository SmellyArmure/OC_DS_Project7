''' 
-*- coding: utf-8 -*-
To run from the directory 'WEB':
python api/api_flask.py
'''

# Load librairies
import sys
import dill
import pandas as pd
import sklearn
from flask import Flask, jsonify, request
import json
from sklearn.neighbors import NearestNeighbors
# import the CustTransformer class (one step of the best_model pipeline)
from api_functions import CustTransformer

# Loading data and model (all the files are in WEB/data)
#--------------
# description of each feature
feat_desc = pd.read_csv("data/feat_desc.csv",
					    index_col=0)
# # cleaned data to apply the best pipeline
# with open('data\\X_test_cleaned.pkl', 'rb') as file:
#     X_test = dill.load(file)
# cleaned data to apply the best pipeline
with open('data\\dict_cleaned_samp.pkl', 'rb') as file:
    dict_cleaned = dill.load(file)
# best model and best threshold
with open('model\\bestmodel_thresh.pkl', 'rb') as file:
    best_model, thresh = dill.load(file)

# compute processed data (first steps of the best_model pipeline)
X_train = dict_cleaned['X_train']
y_train = dict_cleaned['y_train']
X_test = dict_cleaned['X_test']
# split the steps of the best pipeline
preproc_step = best_model.named_steps['preproc']
featsel_step = best_model.named_steps['featsel']
clf_step = best_model.named_steps['clf']
# compute the preprocessed data (encoding and standardization)
X_tr_prepro = preproc_step.transform(X_train)
X_te_prepro = preproc_step.transform(X_test)
# get the name of the columns after encoding
preproc_cols = X_tr_prepro.columns
# get the name of the columns selected using SelectFromModel
featsel_cols = preproc_cols[featsel_step.get_support()]
# compute the data to be used by the best classifier
X_tr_featsel = X_tr_prepro[featsel_cols]
X_te_featsel = X_te_prepro[featsel_cols]

###############################################################
# instantiate Flask object
app = Flask(__name__)

# view when API is launched
# Test : http://127.0.0.1:5000
@app.route("/")
def index():
    return "API, models and data loaded…\nwaiting for a request"

# return json object of feature description when needed
# Test : http://127.0.0.1:5000/api/feat_desc
@app.route('/api/feat_desc/')
def send_feat_desc():
    # Convert pd.Series to JSON
    features_desc_json = json.loads(feat_desc.to_json())
    # Return the processed data as a json object
    return jsonify({'status': 'ok',
    		        'data': features_desc_json})

# answer when asking for sk_ids
# Test : http://127.0.0.1:5000/api/sk_ids/
@app.route('/api/sk_ids/')
def sk_ids():
    # Extract list of 50 first 'SK_ID_CURR' ids in the X_test dataframe
    sk_ids = list(X_test.index.sort_values())
    # Returning the processed data
    return jsonify({'status': 'ok',
    		        'data': sk_ids})

# answer when asking for score and decision about one customer
# Test : http://127.0.0.1:5000/api/scoring_cust/?SK_ID_CURR=100128
@app.route('/api/scoring_cust/')
def scoring_cust():
    # Parse http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the data for the customer (pd.DataFrame)
    X_cust = X_test.loc[sk_id_cust:sk_id_cust]
	# Compute the score of the customer (using the whole pipeline)   
    score_cust = best_model.predict_proba(X_cust)[:,1][0]
    # Compute decision according to the best threshold (True: loan refused)
    bool_cust = (score_cust >= thresh)
    # Return processed data
    return jsonify({'status': 'ok',
    		        'SK_ID_CURR': sk_id_cust,
    		        'score': score_cust,
    		        'bool': str(bool_cust)})

# return data of one customer when requested (SK_ID_CURR)
# Test : http://127.0.0.1:5000/api/data_cust/?SK_ID_CURR=100128
@app.route('/api/data_cust/')
def data_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # Get the personal data for the customer (pd.Series)
    X_cust_ser = X_test.loc[sk_id_cust, :]
    X_cust_proc_ser = X_te_featsel.loc[sk_id_cust, :]
    # Convert the pd.Series (df row) of customer's data to JSON
    X_cust_json = json.loads(X_cust_ser.to_json())
    X_cust_proc_json = json.loads(X_cust_proc_ser.to_json())
    # Return the cleaned data
    return jsonify({'status': 'ok',
    				'data': X_cust_json,
    				'data_proc': X_cust_proc_json})

# return data of 20 neighbors of one customer when requested (SK_ID_CURR)
# Test : http://127.0.0.1:5000/api/neigh_cust/?SK_ID_CURR=100128
@app.route('/api/neigh_cust/')
def neigh_cust():
    # Parse the http request to get arguments (sk_id_cust)
    sk_id_cust = int(request.args.get('SK_ID_CURR'))
    # get data of 20 nearest neigh in the X_tr_featsel dataframe (pd.DataFrame)
    neigh = NearestNeighbors(n_neighbors=20)
    neigh.fit(X_tr_featsel)
    X_cust = X_te_featsel.loc[sk_id_cust: sk_id_cust]
    idx = neigh.kneighbors(X=X_cust,
                           n_neighbors=20,
                           return_distance=False).ravel()
    nearest_cust_idx = list(X_tr_featsel.iloc[idx].index)
    X_neigh_df = X_tr_featsel.loc[nearest_cust_idx, :]
    y_neigh = y_train.loc[nearest_cust_idx]
    # Convert the pd.DataFrame (20 df rows) of customer's data to JSON
    X_neigh_json = json.loads(X_neigh_df.to_json())
    y_neigh_json = json.loads(y_neigh.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_neigh': X_neigh_json,
    				'y_neigh': y_neigh_json})

# return all data of training set when requested
# Test : http://127.0.0.1:5000/api/all_proc_data/
@app.route('/api/all_proc_data_tr/')
def all_proc_data_tr():
    # get all data from X_tr_featsel, X_te_featsel and y_train data
    # and convert the pd.DataFrame (data to JSON
    X_tr_featsel_json = json.loads(X_tr_featsel.to_json())
    y_train_json = json.loads(y_train.to_json())
    # Return the cleaned data jsonified
    return jsonify({'status': 'ok',
    				'X_tr_proc': X_tr_featsel_json,
    				'y_train': y_train_json})

# @app.route('/api/shap_values/')
# # Test : http://127.0.0.1:5000/api/shap_values
# def get_shap_values():

#     # Converting the pd.Series to JSON
#     XXXXXX_json = json.loads(XXXXXXXX.to_json())
#     # Returning the processed data
#     return jsonify({
#         'status': 'ok',
#         'data': XXXXXX_json})

# @app.route('/api/shap_values/')
# # Test : http://127.0.0.1:5000/api/shap_values
# def get_shap_values():

#     # Converting the pd.Series to JSON
#     XXXXXX_json = json.loads(XXXXXXXX.to_json())
#     # Returning the processed data
#     return jsonify({
#         'status': 'ok',
#         'data': XXXXXX_json})


####################################
# if the api is run and not imported as a module
if __name__ == "__main__":
    app.run(debug=True)
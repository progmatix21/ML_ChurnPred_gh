#!/usr/bin/env python3
# coding: utf-8
'''
    churn_ann_deploy.py, an application to deploy a serialized Keras ANN 
    for a specific dataset.
    Copyright (C) 2021  Atrij Talgery(github/progmatix21)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''
# # Demo of using ANN and Keras for Churn Prediction
# 
# ### However we use a saved ANN model to do our prediction as a precursor to deployment.

import pandas as pd
import io
from flask import Flask, render_template, request, Response
app = Flask(__name__, template_folder = ".")

@app.route("/")
def hello():
    hello_str = '''<h1>Deployment of Churn ANN model</h1>
    Go to <a href = /upload>/upload</a> to upload data in CSV format. '''
    return(hello_str)
   
@app.route('/upload')
def upload_file():
    return render_template('./upload.html')

@app.route('/predict', methods = ['GET','POST'])
def file_uploader():
    if request.method == 'POST':
        f = request.files['file'] #f is a file-like object
        churndf = pd.read_csv(f)
        predstr = ann_churn_predict(churndf).to_csv(index=False)

        return Response(predstr, mimetype='text/plain')

def ann_churn_predict(testdf):
    '''Takes in test dataframe and returns predictions dataframe with customer-identifiable key field. '''
    
    churndf = testdf
    #Separate out customer identifiable info to merge with the result.
    Xmeta = churndf.iloc[:,:3]
    #Separate into data and target
    X_test = churndf.iloc[:,3:-1] #Skip the first three metadata columns and the last target column.
    #For testing data y is a dummy column and is not used but needs to be present.
    y = churndf.iloc[:,-1:]

    # We dont need feature encoding or scaling because we load our pickled column transformer 
    # that we trained on the original dataset.

    import pickle
    colxformer = pickle.load(open("colxformer.pkl","rb"))

    #Encode and scale the test data.
    X_test_es = pd.DataFrame(colxformer.transform(X_test))

    # We will load our saved ANN which is already trained on customer churn data.

    from keras.models import load_model
    model = load_model('./churn_ann_model.h5')

    # We already have X_test_es as the encoded and scaled test set. We will use it for the prediction.
    threshold = 0.5
    yhat = (model.predict(X_test_es)>threshold).astype("int32")
    #For sigmoid activation, probability of >0.5 signifies churn or value of 1.
    #Convert yhat to a dataframe.
    yhatdf = pd.DataFrame({'Churn':pd.Series(yhat.ravel())})
    #Now prepare the dataframe and csv to return.
    churn_pred = pd.concat([Xmeta,yhatdf],axis = 1)
    return(churn_pred)

if __name__ == "__main__":
    app.run()
    exit()


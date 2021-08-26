# Churn Prediction using ANN with deployment using Flask

This is a project to predict customer-churn using Keras/ANN and later Flask to deploy the model.

We use a two step process. In the first step, we use a Jupyter Lab notebook for data exploration and to experiment with and tune hyperparameters for the ANN as well as data transformation(encoding/scaling). In this step, we also export our models as a preparation for deployment. 

In the second step, we use the stored/serialized models in a Flask app for deployment.

## Data preparation and encoding

The dataset is in CSV format with 10000 records. There are no missing values.

An exploratory data analysis using Seaborn, clearly shows us the categorical and continuous variables. The features:Geography, Gender. etc. show clear imbalances and hence are a basis for stratification. We use a sklearn column transformer for a one-step encoding/transformation.
This gives us both convenience and ease of deployment. Other details of the data prep step are available in the notebook.

## Training the model

The ANN is built with Keras and trained experimentally using various configurations keeping track of loss and accuracy. We find a best-case accuracy of about 85% with a recall and precision of 50% and 66% respectively attained after just about 10 epochs. 

## Deploying the model

Model is deployed using Flask. Test data is uploaded as a CSV and the response is a CSV dump. 

## Observations

Experiments with various configurations and stratification do not seem to give us an accuracy better than 85%. A precision-recall tradeoff is possible by adjusting the probability threshold. 


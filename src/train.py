import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import mlflow
from mlflow.models import infer_signature
import os
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse
load_dotenv()


def hyperparameter_tuning(x_train,y_train,param_grid):
    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(x_train,y_train)
    return grid_search

# Load the parameters from params.yaml

params = yaml.safe_load(open("params.yaml"))['train']

def train(data_path,model_path,random_state,n_estimators,max_depth):
    data=pd.read_csv(data_path)
    x = data.drop(columns=["Outcome"])
    y=data["Outcome"]

    mlflow.set_tracking_uri(os.getenv['MLFLOW_TRACKING_URI'])

    # starat the MLFLOW
    with mlflow.start_run():
        # split the dataset 
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
        signature = infer_signature(x_train,y_train)

        # Define hyperparameter grid

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5,10,None],
            'min_samples_split': [2,5],
            'min_samples_leaf': [1,2]
        }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(x_train,y_train,param_grid)

        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test,y_pred)
        print(f"Accuracy: {accuracy}")
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv()

params = yaml.safe_load(open('params.yaml'))['train']

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    x = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    # loading the model from the disk
    model = pickle.load(open(model_path,'rb'))

    predictions = model.predict(x)
    accuracy = accuracy_score(y,predictions)

    # log metrics
    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy: {accuracy}")

if __name__=='__main__':
    data_path = params['data']
    model_path = params['model']
    evaluate(data_path,model_path)

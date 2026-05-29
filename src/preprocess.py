import pandas as pd
import sys
import yaml
import os

# Load parameters from params.yml
params = yaml.safe_load(open("params.yml"))["preprocess"]

def preprocess(input_path,output_path):
    df = pd.read_csv(input_path, header=None)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    df.to_csv(output_path,header=None, index=False)
    print(f"Preprocess data saved to {output_path}")

if __name__=="__main__":
    preprocess(params["input"],params["output"])
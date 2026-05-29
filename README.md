# Diabetes Classification using MLOps

An end-to-end Machine Learning project for diabetes classification built with modern MLOps practices. This project demonstrates experiment tracking, data versioning, and reproducible ML pipelines using DVC, MLflow, and DagsHub.

---

## Project Overview

The goal of this project is to build a machine learning pipeline capable of predicting whether a patient is likely to have diabetes based on medical attributes.

The project focuses not only on model development but also on implementing production-oriented MLOps workflows such as:

- Data Version Control (DVC)
- Experiment Tracking (MLflow)
- Remote Collaboration (DagsHub)
- Reproducible Pipelines
- Modular Project Structure

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NumPy
- MLflow
- DVC
- DagsHub
- Git & GitHub

---

## Project Workflow

1. Data Ingestion
2. Data Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Experiment Tracking with MLflow
7. Data & Pipeline Versioning with DVC
8. Remote Repository Integration with DagsHub

---

## Project Structure

```bash
├── data/
├── src/
├── models/
├── dvc.yaml
├── params.yaml
├── requirements.txt
└── README.md

## DVC Pipeline Stages

This project uses DVC to create a reproducible machine learning pipeline.

Each stage defines:

* dependencies
* parameters
* outputs
* execution commands

---

# 1. Preprocessing Stage

```bash id="uy0z76"
dvc stage add -n preprocess \
    -p preprocess.input,preprocess.output \
    -d src/preprocess.py -d data/raw/data.csv \
    -o data/preprocessed/data.csv \
    python src/preprocess.py
```

### Purpose

This stage preprocesses the raw dataset and generates cleaned/transformed data for training.

### Components

* `-n preprocess`
  Creates a stage named `preprocess`.

* `-p preprocess.input,preprocess.output`
  Tracks parameters from `params.yaml`.

* `-d src/preprocess.py`
  Tracks the preprocessing script as a dependency.

* `-d data/raw/data.csv`
  Tracks the raw dataset dependency.

* `-o data/preprocessed/data.csv`
  Defines the generated output file.

* `python src/preprocess.py`
  Command executed by DVC.

---

# 2. Training Stage

```bash id="0d76gu"
dvc stage add -n train \
    -p train.data,train.model,train.random_state,train.n_estimators,train.max_depth \
    -d src/train.py -d data/preprocessed/data.csv \
    -o models/random_forest.pkl \
    python src/train.py
```

### Purpose

This stage trains the machine learning model using the preprocessed dataset.

### Components

* `-n train`
  Creates a stage named `train`.

* `-p train.data,train.model,train.random_state,train.n_estimators,train.max_depth`
  Tracks training-related parameters from `params.yaml`.

* `-d src/train.py`
  Tracks the training script dependency.

* `-d data/preprocessed/data.csv`
  Uses preprocessed data as input.

* `-o models/random_forest.pkl`
  Stores the trained model artifact.

* `python src/train.py`
  Executes the training pipeline.

---

# 3. Evaluation Stage

```bash id="zgq67p"
dvc stage add -n evaluate \
    -d src/evaluate.py -d models/model.pkl -d data/raw/data.csv \
    python src/evaluate.py
```

### Purpose

This stage evaluates the trained model on the dataset and generates evaluation metrics.

### Components

* `-n evaluate`
  Creates a stage named `evaluate`.

* `-d src/evaluate.py`
  Tracks the evaluation script dependency.

* `-d models/model.pkl`
  Uses the trained model for evaluation.

* `-d data/raw/data.csv`
  Uses dataset input during evaluation.

* `python src/evaluate.py`
  Executes the evaluation process.

---

# Running the Pipeline

Run the complete pipeline using:

```bash id="97e0ti"
dvc repro
```

DVC automatically:

* detects changed files/parameters
* reruns only affected stages
* maintains reproducibility
* tracks pipeline dependencies

---

# Generated Files

After adding stages, DVC creates:

* `dvc.yaml` → defines the ML pipeline
* `dvc.lock` → stores exact file hashes and reproducible state

These files help ensure consistent and reproducible ML workflows.

## DVC Remote Storage Setup with DagsHub S3

This project uses DVC with DagsHub S3-compatible remote storage to version datasets and model artifacts remotely.

---

# Configure DVC Remote

Add the DVC remote storage:

```bash id="ph9g8x"
dvc remote add origin s3://dvc
```

Configure the DagsHub S3 endpoint:

```bash id="t0avp8"
dvc remote modify origin endpointurl https://dagshub.com/shikharjul01/mlops-diabetes-prediction.s3
```

---

# Configure Authentication Credentials

Set the access key:

```bash id="v0m4vl"
dvc remote modify origin --local access_key_id <YOUR_ACCESS_KEY>
```

Set the secret access key:

```bash id="6r0otv"
dvc remote modify origin --local secret_access_key <YOUR_SECRET_KEY>
```

---

# Why `--local` is Used

The `--local` flag stores credentials inside:

```text id="bktk0p"
.dvc/config.local
```

This file is:

* excluded from Git
* not shared publicly
* safer for handling secrets

This prevents exposing cloud credentials in the repository.

---

# Push Data to Remote Storage

After configuring the remote, upload tracked data and models using:

```bash id="0q8ezs"
dvc push
```

This uploads:

* datasets
* model files
* pipeline outputs

to the configured DagsHub remote storage.

---

# Pull Data from Remote Storage

To download data and artifacts from remote storage:

```bash id="gxk3wn"
dvc pull
```

This allows other users or deployment environments to reproduce the pipeline without manually sharing files.

---

# Benefits of Using DVC Remote Storage

* Version control for large datasets
* Centralized artifact storage
* Reproducible ML pipelines
* Lightweight Git repository
* Easy collaboration across environments

The combination of Git + DVC + DagsHub enables complete reproducibility for machine learning workflows.

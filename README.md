# Airflow Pipeline Setup

This repository contains the necessary setup for running an Apache Airflow pipeline for data preprocessing and training on a local machine for the task of customer churn prediction.

## Requirements

- Python 3.6+
- Apache Airflow 2.9.2
- pandas
- scikit-learn

## Installation

1. Clone this repository:
    ```sh
    git clone https://github.com/sagarikajadon13/airflow-pipeline-setup.git
    cd airflow-pipeline-setup
    ```

2. Create a Conda environment and activate it:
    ```sh
    conda create --name airflow-env python=3.8
    conda activate airflow-env
    ```

3. Install the required packages:
    ```sh
    pip install apache-airflow==2.9.2 pandas scikit-learn
    ```

## Setup Airflow

1. Ensure that `data_pipeline.py` is placed in the `dags` directory specified in your `airflow.cfg`. By default, this is usually the `dags` folder in your Airflow home directory (e.g., `~/airflow/dags`).

2. Initialize the Airflow database:
    ```sh
    airflow db init
    ```

3. Create an Airflow user:
    ```sh
    airflow users create \
        --username admin \
        --firstname FIRST_NAME \
        --lastname LAST_NAME \
        --role Admin \
        --email admin@example.com \
        --password admin
    ```
    This create a user with username as well as password with 'admin'.

4. Start the Airflow webserver:
    ```sh
    airflow webserver --port 8080
    ```

5. Start the Airflow scheduler:
    ```sh
    airflow scheduler
    ```

## Accessing the Airflow UI

Once the webserver is up and running, you can access the Airflow UI by navigating to `http://localhost:8080` in your web browser.

## Report- 

Here is the report containing more details about the EDA, preprocessing and modelling of the data- 
https://docs.google.com/document/d/14RkQ5BxkZd8mSx4UTS-IWGjwaRG7CqwJz71wUUqZ62g/edit#heading=h.e2in7lzhugur


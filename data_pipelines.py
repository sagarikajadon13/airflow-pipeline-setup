from datetime import timedelta, datetime
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def preprocess():
    df = pd.read_excel('./airflow-pipeline-setup/data/Telco_customer_churn.xlsx')

    df = df[df['Total Charges'] != " "].reset_index(drop=True)
    df['Total Charges'] = df['Total Charges'].astype(float)

    enc = LabelEncoder()
    df['Contract'] = enc.fit_transform(df['Contract'])
    df['Internet Service'] = enc.fit_transform(df['Internet Service'])
    df['Multiple Lines'] = enc.fit_transform(df['Multiple Lines'])
    df['Payment Method'] = enc.fit_transform(df['Payment Method'])
    df['Tech Support'] = enc.fit_transform(df['Tech Support'])

    df_no = df[df['Churn Value'] == 0].sample(n=1869, random_state=42).reset_index(drop=True)
    df_no_test = df[df['CustomerID'].isin(df_no['CustomerID']) == False].reset_index(drop=True)

    df_yes = df[df['Churn Value'] == 1].reset_index(drop=True)
    df = pd.concat([df_no, df_yes], axis=0).reset_index(drop=True)

    df = df[['Tenure Months', 'Tech Support', 'Internet Service', 'Payment Method', 'Contract', 'Total Charges', 'Churn Value']]
    df_no_test = df_no_test[['Tenure Months', 'Tech Support', 'Internet Service', 'Payment Method', 'Contract', 'Total Charges', 'Churn Value']] 

    df.to_csv('./airflow-pipeline-setup/data/trainval.csv')
    df_no_test.to_csv('./airflow-pipeline-setup/data/bench_no.csv')

def train():
    trainval = pd.read_csv('./airflow-pipeline-setup/data/trainval.csv')
    bench_no = pd.read_csv('./airflow-pipeline-setup/data/bench_no.csv')

    y_bench_no = bench_no['Churn Value']
    X_bench_no = bench_no.drop('Churn Value', axis=1)

    y = trainval['Churn Value']
    X = trainval.drop('Churn Value', axis=1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state=42, stratify=y)

    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    X_val = X_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    clf = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=4)
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    print(accuracy_score(y_val, val_preds))

    bench_no_preds = clf.predict(X_bench_no)
    print(accuracy_score(y_bench_no, bench_no_preds))

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'data_processing_and_model_training',
    default_args=default_args,
    description='A DAG for data preprocessing and training',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2023, 6, 1),  
    tags=['example'],
)

t1 = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess,
    dag=dag,
)

t2 = PythonOperator(
    task_id='train',
    python_callable=train,
    dag=dag,
)

t1 >> t2

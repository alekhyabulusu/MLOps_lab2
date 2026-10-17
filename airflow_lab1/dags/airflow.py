# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
#from airflow.providers.standard.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, validate_data, data_preprocessing, build_save_model, load_model_elbow

# Define default arguments for your DAG
default_args = {
    'owner': 'Alekhya Bulusu',
    'start_date': datetime(2026, 10, 17),
    'retries': 1,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# DAG instance named 'Airflow_Lab1' with the defined default arguments
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Automated customer segmentation using K-Mean',
    schedule_interval=None,
    catchup=False, 
) as dag:

    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task to validate data, depends on 'load_data_task'
    validate_data_task = PythonOperator(
        task_id="validate_data_task",
        python_callable=validate_data,
        op_args=[load_data_task.output],
    )

    # Task to perform data preprocessing, depends on 'validate_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[validate_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "customer_kmeans.sav"],
    )

    # Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=[data_preprocessing_task.output, build_save_model_task.output],
    )

    # Set task dependencies
    load_data_task >> validate_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()

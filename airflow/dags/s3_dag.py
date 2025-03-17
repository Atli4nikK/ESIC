import os
import sys
from datetime import datetime

# Добавляем путь к общим утилитам
sys.path.append('/opt/airflow')
from common.utils import get_default_args, get_project_path, get_project_config

from airflow import DAG
from airflow.operators.python import PythonOperator

from datetime import datetime,timedelta
from airflow.utils.dates import days_ago

from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount

from common.notify import notify_on_failure, notify_on_success
from common.yandex_downloader import yandex_downloader
from common.upload_to_s3 import upload_to_s3


# Очищает папку с изображениями
def clean_local_files(**kwargs):
    
    import shutil

    keyword = kwargs['keyword']
    local_dir = f'/opt/airflow/dags/esic/data/{keyword}'

    shutil.rmtree(local_dir)

default_args = {
    "depends_on_past": False,
    "email": ["airflow@example.com"],
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    'on_failure_callback': notify_on_failure, # or list of functions
    # 'on_success_callback': some_other_function, # or list of functions
    # 'on_retry_callback': another_function, # or list of functions
    # 'sla_miss_callback': yet_another_function, # or list of functions
    # 'on_skipped_callback': another_function, #or list of functions
    # 'trigger_rule': 'all_success'
}

with DAG(
    dag_id="s3_dag", 
    default_args=default_args,
    description="s3 DAG",
    catchup=False,
    schedule=None,
    start_date=days_ago(2),
    tags=["koldyrkaevs", "esic", "s3"],
) as dag:
    
    categories = {
        'машина скорой помощи России': 'dataset/ambulance',
        'полицейская машина России': 'dataset/police',
        'пожарная машина России': 'dataset/fire',
    }

    upload_tasks = []
    # Создаем список для хранения задач
    tasks_data = []
    tasks_check = []
    clean_tasks = []
    for keyword, s3_dir in categories.items():
        create_dataset = PythonOperator(
            task_id=f"create_dataset_{keyword.replace(' ', '_')}",
            python_callable=yandex_downloader,
            op_kwargs={
                'keyword': keyword,
                's3_dir': s3_dir,
                'bucket_name': 'koldyrkaevs3',
                'num_images': 1500,
            }
        )
        tasks_data.append(create_dataset)  # Добавляем задачу в список

        train_model = DockerOperator(
            task_id=f"car_checker_{keyword.replace(' ', '_')}",
            docker_url="tcp://docker-socket-proxy:2375",
            api_version="auto",
            auto_remove="force", # в случае True контейнер самовыпиливается после отработки ДАГА
            image="car-filtering:latest",
            container_name="car-filtering",
            environment={},
            mounts=[
                Mount(
                    source=f"C:/Users/koldyrkaev/Desktop/GIT/ESIC/airflow/dags/data/{keyword}",
                    target="/app/images",
                    type="bind"
                ),
                Mount(
                    source=f"C:/Users/koldyrkaev/Desktop/GIT/ESIC/train_model/no_cars/{keyword}",
                    target="/app/no_cars",
                    type="bind"
                ),
                Mount(
                    source="C:/Users/koldyrkaev/Desktop/GIT/ESIC/train_model/check.txt",
                    target="/app/check.txt",
                    type="bind"
                ),
                ],
                #network_mode="bridge",
                #docker run -it --rm -v <path_to_your_folder>:/app/images -v <path_to_your_folder>:/app/no_cars -v <path_to_your_check.txt>:/app/check.txt car-filtering
                #command=["python", "main_extend.py"], # если нам надо запустить другой скрипт без пересборки образа
                #docker run -it --rm -v  -v  -v  car-filtering
        )

        tasks_check.append(train_model)
        
        upload_to_s3_task = PythonOperator(
            task_id=f"upload_to_s3_{keyword.replace(' ', '_')}",
            python_callable=upload_to_s3,
            op_kwargs={
                'keyword': keyword,
                's3_dir': s3_dir,
                'bucket_name': 'koldyrkaevs3',
            }
        )
            
        upload_tasks.append(upload_to_s3_task)

        clean_local_files_task = PythonOperator(
            task_id=f"clean_local_files_{keyword.replace(' ', '_')}",
            python_callable=clean_local_files,
            op_kwargs={
                'keyword': keyword,
            }
        )
        clean_tasks.append(clean_local_files_task)

    # Устанавливаем последовательность выполнения
    for i in range(1, len(tasks_data)):
        tasks_data[i-1] >> tasks_check[i-1] >> upload_tasks[i-1] >> clean_tasks[i-1] >> tasks_data[i] >> tasks_check[i] >> upload_tasks[i] >> clean_tasks[i]  # Каждая предыдущая задача становится предком следующей
    
    success = PythonOperator(
            task_id="success",
            python_callable=notify_on_success,
        )
    clean_tasks[-1] >> success

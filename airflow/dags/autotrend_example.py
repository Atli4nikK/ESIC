"""
DAG для проекта AutoTrend
"""
import os
import sys
from datetime import datetime

# Добавляем путь к общим утилитам
sys.path.append('/opt/airflow')
from common.utils import get_default_args, get_project_path, get_project_config

# Импорты Airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# --- КОНФИГУРАЦИЯ ПРОЕКТА ---
PROJECT_NAME = "autotrend"
DAG_ID = f"{PROJECT_NAME}_example_dag"
DAG_DESCRIPTION = f"Пример DAG для проекта {PROJECT_NAME}"
DAG_SCHEDULE = "@daily"
DAG_CATCHUP = False
DAG_TAGS = [PROJECT_NAME]

# Получаем путь к проекту
PROJECT_PATH = get_project_path(PROJECT_NAME)

# Конфигурация проекта из переменных Airflow
PROJECT_CONFIG = get_project_config(PROJECT_NAME)


# --- ОПРЕДЕЛЕНИЕ ФУНКЦИЙ ---
def example_task(**context):
    """
    Пример задачи для проекта AutoTrend
    
    Args:
        **context: Контекст выполнения задачи Airflow
        
    Returns:
        str: Результат выполнения задачи
    """
    print(f"Выполняется задача для проекта {PROJECT_NAME}")
    print(f"Путь к проекту: {PROJECT_PATH}")
    print(f"Дата выполнения: {context['execution_date']}")
    
    # Пример использования XComs для передачи данных между задачами
    context['ti'].xcom_push(key=f'{PROJECT_NAME}_task_status', value='completed')
    
    return "Задача успешно выполнена"


# --- ОПРЕДЕЛЕНИЕ DAG ---
with DAG(
    DAG_ID,
    default_args=get_default_args(owner=PROJECT_NAME),
    description=DAG_DESCRIPTION,
    schedule_interval=DAG_SCHEDULE,
    catchup=DAG_CATCHUP,
    tags=DAG_TAGS,
    max_active_runs=1
) as dag:
    
    # Определение задач
    task1 = PythonOperator(
        task_id='example_task',
        python_callable=example_task,
        provide_context=True,
        doc_md="""
        ## Пример задачи AutoTrend
        
        Эта задача демонстрирует базовую структуру PythonOperator.
        
        **Функциональность**:
        * Выводит информацию о проекте AutoTrend
        * Демонстрирует использование XComs
        """,
    )
    
    # Определение порядка выполнения задач
    task1 
import argparse
import time

from utils.preprocessing import Preprocessing
import json
import psycopg2


def get_task_id_list(task_type):
    conn = psycopg2.connect(
        dbname=config_data['database']['database'],
        user=config_data['database']['user'],
        password=config_data['database']['password'],
        host=config_data['database']['host'],
        port=config_data['database']['port']
    )
    cursor = conn.cursor()
    cursor.execute('SET search_path TO public')
    cursor.execute("SELECT current_schema()")
    cursor.execute(
        "SELECT id, creator, task_id_ref FROM avt_task WHERE task_type = %s and task_stat < 0 ORDER BY task_stat DESC",
        (task_type,))
    result = cursor.fetchall()
    return [[res[0], res[1], res[2]] for res in result]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--avt_task_id', type=int, required=True, help='task id')
    parser.add_argument('--config_file', type=str, required=True, help='config file')
    args = parser.parse_args()
    while True:
        task_type = 2
        config_data = json.load(open(args.config_file))
        list_task = get_task_id_list(task_type)
        # print(list_task)
        if len(list_task) > 0:
            for task_id, creator, task_id_ref in list_task:
                if creator == "system":
                    if task_id_ref == 0 or task_id_ref is None:
                        preprocessing = Preprocessing()
                        preprocessing.automatic_process(task_id, config_data)
                    else:
                        pass
                else:
                    preprocessing = Preprocessing()
                    preprocessing.process(task_id, config_data)
        time.sleep(5)

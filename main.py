import argparse
from utils.preprocessing import Preprocessing
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--avt_task_id', type=int, required=True, help='task id')
    parser.add_argument('--config_file', type=str, required=True, help='config file')
    args = parser.parse_args()
    config_data = json.load(open(args.config_file))
    preprocessing = Preprocessing()
    preprocessing.process(args.avt_task_id, config_data)

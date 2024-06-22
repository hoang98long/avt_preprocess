import argparse
from utils.preprocessing import Preprocessing


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--avt_task_id', type=int, required=True, help='task id')
    args = parser.parse_args()
    preprocessing = Preprocessing()
    preprocessing.process(args.avt_task_id)

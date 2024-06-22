import ftplib
import os.path
from utils.config import *
from utils.preprocessing import Preprocessing
import psycopg2
import json
import argparse


# import ast


def connect_ftp():
    ftp = ftplib.FTP()
    ftp.connect(FTP_HOST, FTP_PORT)
    ftp.set_pasv(True)
    ftp.login(user=FTP_USERNAME, passwd=FTP_PASSWORD)
    return ftp


def check_and_create_directory(ftp, directory):
    try:
        ftp.cwd(directory)
    except ftplib.error_perm as e:
        if str(e).startswith('550'):
            ftp.mkd(directory)
        else:
            print(f"Error changing to directory '{directory}': {e}")


def download_file(ftp, ftp_file_path, local_file_path):
    try:
        with open(local_file_path, 'wb') as file:
            ftp.retrbinary(f"RETR {ftp_file_path}", file.write)
        print(f"Downloaded '{ftp_file_path}' to '{local_file_path}'")
    except ftplib.all_errors as e:
        print(f"FTP error: {e}")


def merge_channel(conn, task_id, task_param):
    input_file = task_param['input_file']
    single_bands = task_param['single_bands']
    # single_bands = ast.literal_eval(single_bands)
    multi_bands = task_param['multi-bands']
    # multi_bands = ast.literal_eval(multi_bands)
    try:
        filename = input_file.split("/")[-1]
        local_file_path = LOCAL_SRC_MERGE_IMAGE_PATH + filename
        ftp = connect_ftp()
        download_file(ftp, input_file, local_file_path)
        preprocess_image = Preprocessing()
        result_image_path = preprocess_image.merge_image(local_file_path, single_bands, multi_bands)
        result_image_name = result_image_path.split("/")[-1]
        ftp_dir = os.path.join(FTP_MERGE_IMAGE_PATH, result_image_name.split(".")[0])
        check_and_create_directory(ftp, ftp_dir)
        ftp.cwd(str(ftp_dir))
        export_types = ["png", "jpg", "tiff"]
        for export_type in export_types:
            filename = result_image_name + "." + export_type
            with open(result_image_path, "rb") as file:
                save_dir = ftp_dir + "/" + filename
                ftp.storbinary(f"STOR {save_dir}", file)
        print("Connection closed")
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'finished', task_output = %s WHERE task_id = %s", (ftp_dir, task_id,))
    except ftplib.all_errors as e:
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'error' WHERE task_id = %s", (task_id,))
        print(f"FTP error: {e}")


def sharpen_image(conn, task_id, task_param):
    ORG_input_file = task_param['input_file']
    PAN_input_file = task_param['input_file_single_band']
    try:
        org_filename = ORG_input_file.split("/")[-1]
        local_org_file_path = LOCAL_SRC_SHARPEN_IMAGE_PATH + org_filename
        pan_filename = PAN_input_file.split("/")[-1]
        local_pan_file_path = LOCAL_SRC_SHARPEN_IMAGE_PATH + pan_filename
        ftp = connect_ftp()
        download_file(ftp, ORG_input_file, local_org_file_path)
        download_file(ftp, PAN_input_file, local_pan_file_path)
        preprocess_image = Preprocessing()
        result_image_path = preprocess_image.sharpen_image(local_org_file_path, local_pan_file_path)
        result_image_name = result_image_path.split("/")[-1]
        ftp_dir = FTP_SHARPEN_IMAGE_PATH
        ftp.cwd(str(ftp_dir))
        save_dir = ftp_dir + "/" + result_image_name
        with open(result_image_path, "rb") as file:
            ftp.storbinary(f"STOR {save_dir}", file)
        print("Connection closed")
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'finished', task_output = %s WHERE task_id = %s", (save_dir, task_id,))
    except ftplib.all_errors as e:
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'error' WHERE task_id = %s", (task_id,))
        print(f"FTP error: {e}")


def adjust_gamma(conn, task_id, task_param):
    src_img_path = task_param['input_file']
    gamma = task_param['gamma']
    try:
        filename = src_img_path.split("/")[-1]
        local_file_path = LOCAL_SRC_ADJUST_IMAGE_PATH + filename
        ftp = connect_ftp()
        download_file(ftp, src_img_path, local_file_path)
        preprocess_image = Preprocessing()
        result_image_path = preprocess_image.adjust_gamma(local_file_path, gamma)
        result_image_name = result_image_path.split("/")[-1]
        ftp_dir = FTP_ADJUST_IMAGE_PATH
        ftp.cwd(str(ftp_dir))
        save_dir = ftp_dir + "/" + result_image_name
        with open(result_image_path, "rb") as file:
            ftp.storbinary(f"STOR {save_dir}", file)
        print("Connection closed")
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'finished', task_output = %s WHERE task_id = %s", (save_dir, task_id,))
    except ftplib.all_errors as e:
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'error' WHERE task_id = %s", (task_id,))
        print(f"FTP error: {e}")


def equalize_hist(conn, task_id, task_param):
    src_img_path = task_param['input_file']
    mode = task_param['mode']
    tileGridSize = task_param['tileGridSize']
    try:
        filename = src_img_path.split("/")[-1]
        local_file_path = LOCAL_SRC_EQUALIZE_IMAGE_PATH + filename
        ftp = connect_ftp()
        download_file(ftp, src_img_path, local_file_path)
        preprocess_image = Preprocessing()
        result_image_path = preprocess_image.hist_equalize(local_file_path, mode, tileGridSize)
        result_image_name = result_image_path.split("/")[-1]
        ftp_dir = FTP_EQUALIZE_IMAGE_PATH
        ftp.cwd(str(ftp_dir))
        save_dir = ftp_dir + "/" + result_image_name
        with open(result_image_path, "rb") as file:
            ftp.storbinary(f"STOR {save_dir}", file)
        print("Connection closed")
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'finished', task_output = %s WHERE task_id = %s", (save_dir, task_id,))
    except ftplib.all_errors as e:
        cursor = conn.cursor()
        cursor.execute("UPDATE avt_tasks SET task_stat = 'error' WHERE task_id = %s", (task_id,))
        print(f"FTP error: {e}")


def process(task_id):
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM avt_tasks WHERE task_id = %s", (task_id,))
    result = cursor.fetchone()
    print(result)
    # task_param = json.loads(result[3])
    # algorithm = task_param["algorithm"]
    # if algorithm == "merge":
    #     merge_channel(conn, task_id, task_param)
    # elif algorithm == "sharpen":
    #     sharpen_image(conn, task_id, task_param)
    # elif algorithm == "adjust":
    #     adjust_gamma(conn, task_id, task_param)
    # elif algorithm == "equalize":
    #     equalize_hist(conn, task_id, task_param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--avt_task_id', type=int, required=True, help='task id')
    args = parser.parse_args()
    process(args.avt_task_id)

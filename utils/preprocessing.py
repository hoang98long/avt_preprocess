import ftplib
import os.path
from utils.config import *
from utils.preprocessing_image import Preprocessing_Image
import psycopg2
import json
import ast
from datetime import datetime

ftp_directory = json.load(open("ftp_directory.json"))
FTP_MERGE_IMAGE_PATH = ftp_directory['merge_image_result_directory']
FTP_SHARPEN_IMAGE_PATH = ftp_directory['sharpen_image_result_directory']
FTP_ADJUST_IMAGE_PATH = ftp_directory['adjust_image_result_directory']
FTP_EQUALIZE_IMAGE_PATH = ftp_directory['equalize_image_result_directory']


def connect_ftp(config_data):
    ftp = ftplib.FTP()
    ftp.connect(config_data['ftp']['host'], config_data['ftp']['port'])
    ftp.set_pasv(True)
    ftp.login(user=config_data['ftp']['user'], passwd=config_data['ftp']['password'])
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


def route_to_db(cursor):
    cursor.execute('SET search_path TO public')
    cursor.execute("SELECT current_schema()")


def get_time():
    now = datetime.now()
    current_datetime = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
    return current_datetime


class Preprocessing:
    def __init__(self):
        pass

    def merge_channel(self, conn, id, task_param, ftp):
        input_file = task_param['input_file']
        single_bands = task_param['single_bands']
        single_bands = ast.literal_eval(single_bands)
        multi_bands = task_param['multi_bands']
        multi_bands = ast.literal_eval(multi_bands)
        try:
            filename = input_file.split("/")[-1]
            local_file_path = LOCAL_SRC_MERGE_IMAGE_PATH + filename
            download_file(ftp, input_file, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.merge_image(local_file_path, single_bands, multi_bands)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = os.path.join(FTP_MERGE_IMAGE_PATH, result_image_name.split(".")[0])
            check_and_create_directory(ftp, ftp_dir)
            ftp.cwd(str(ftp_dir))
            export_types = ["png", "jpg", "tiff"]
            task_output = str({
                "output_dir": ftp_dir
            })
            print(task_output)
            for export_type in export_types:
                filename = result_image_name + "." + export_type
                with open(result_image_path + "." + export_type, "rb") as file:
                    save_dir = ftp_dir + "/" + filename
                    ftp.storbinary(f"STOR {save_dir}", file)
            print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s", (task_output, get_time(), id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def sharpen_image(self, conn, id, task_param, ftp):
        ORG_input_file = task_param['input_file']
        PAN_input_file = task_param['input_file_single_band']
        try:
            org_filename = ORG_input_file.split("/")[-1]
            local_org_file_path = LOCAL_SRC_SHARPEN_IMAGE_PATH + org_filename
            pan_filename = PAN_input_file.split("/")[-1]
            local_pan_file_path = LOCAL_SRC_SHARPEN_IMAGE_PATH + pan_filename
            download_file(ftp, ORG_input_file, local_org_file_path)
            download_file(ftp, PAN_input_file, local_pan_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.sharpen_image(local_org_file_path, local_pan_file_path)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_SHARPEN_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": save_dir
            })
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s", (task_output, get_time(), id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def adjust_gamma(self, conn, id, task_param, ftp):
        src_img_path = task_param['input_file']
        gamma = task_param['gamma']
        try:
            filename = src_img_path.split("/")[-1]
            local_file_path = LOCAL_SRC_ADJUST_IMAGE_PATH + filename
            download_file(ftp, src_img_path, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.adjust_gamma(local_file_path, gamma)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_ADJUST_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": save_dir
            })
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s", (task_output, get_time(), id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def equalize_hist(self, conn, id, task_param, ftp):
        src_img_path = task_param['input_file']
        mode = task_param['mode']
        tileGridSize = task_param['tileGridSize']
        try:
            filename = src_img_path.split("/")[-1]
            local_file_path = LOCAL_SRC_EQUALIZE_IMAGE_PATH + filename
            download_file(ftp, src_img_path, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.hist_equalize(local_file_path, mode, tileGridSize)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_EQUALIZE_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": save_dir
            })
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s", (task_output, get_time(), id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def process(self, id, config_data):
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
        cursor.execute("SELECT task_param FROM avt_task WHERE id = %s", (id,))
        result = cursor.fetchone()
        preprocess = Preprocessing()
        task_param = json.loads(result[0])
        algorithm = task_param["algorithm"]
        ftp = connect_ftp(config_data)
        if algorithm == "ket_hop_kenh":
            preprocess.merge_channel(conn, id, task_param, ftp)
        elif algorithm == "lam_sac_net":
            preprocess.sharpen_image(conn, id, task_param, ftp)
        elif algorithm == "dieu_chinh_anh":
            preprocess.adjust_gamma(conn, id, task_param, ftp)
        elif algorithm == "can_bang_anh":
            preprocess.equalize_hist(conn, id, task_param, ftp)
        cursor.close()

import ftplib
import os.path
from utils.config import *
from utils.preprocessing_image import Preprocessing_Image
import psycopg2
import json


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


def route_to_db(cursor):
    cursor.execute('SET search_path TO public')
    cursor.execute("SELECT current_schema()")


class Preprocessing:
    def __init__(self):
        pass

    def merge_channel(self, conn, id, task_param):
        input_file = task_param['input_file']
        single_bands = task_param['single_bands']
        # single_bands = ast.literal_eval(single_bands)
        multi_bands = task_param['multi_bands']
        # multi_bands = ast.literal_eval(multi_bands)
        try:
            filename = input_file.split("/")[-1]
            local_file_path = LOCAL_SRC_MERGE_IMAGE_PATH + filename
            ftp = connect_ftp()
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
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s WHERE id = %s", (task_output, id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def sharpen_image(self, conn, id, task_param):
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
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.sharpen_image(local_org_file_path, local_pan_file_path)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_SHARPEN_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = {
                "output_image": save_dir
            }
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s WHERE id = %s", (task_output, id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def adjust_gamma(self, conn, id, task_param):
        src_img_path = task_param['input_file']
        gamma = task_param['gamma']
        try:
            filename = src_img_path.split("/")[-1]
            local_file_path = LOCAL_SRC_ADJUST_IMAGE_PATH + filename
            ftp = connect_ftp()
            download_file(ftp, src_img_path, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.adjust_gamma(local_file_path, gamma)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_ADJUST_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = {
                "output_image": save_dir
            }
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s WHERE id = %s", (task_output, id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def equalize_hist(self, conn, id, task_param):
        src_img_path = task_param['input_file']
        mode = task_param['mode']
        tileGridSize = task_param['tileGridSize']
        try:
            filename = src_img_path.split("/")[-1]
            local_file_path = LOCAL_SRC_EQUALIZE_IMAGE_PATH + filename
            ftp = connect_ftp()
            download_file(ftp, src_img_path, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.hist_equalize(local_file_path, mode, tileGridSize)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_EQUALIZE_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = {
                "output_image": save_dir
            }
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s WHERE id = %s", (task_output, id,))
            conn.commit()
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            print(f"FTP error: {e}")

    def process(self, id):
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        cursor = conn.cursor()
        cursor.execute('SET search_path TO public')
        cursor.execute("SELECT current_schema()")
        cursor.execute("SELECT * FROM avt_task WHERE id = %s", (id,))
        result = cursor.fetchone()
        preprocess = Preprocessing()
        task_param = json.loads(result[3])
        algorithm = task_param["algorithm"]
        if algorithm == "ket_hop_kenh":
            preprocess.merge_channel(conn, id, task_param)
        elif algorithm == "lam_sac_net":
            preprocess.sharpen_image(conn, id, task_param)
        elif algorithm == "dieu_chinh_anh":
            preprocess.adjust_gamma(conn, id, task_param)
        elif algorithm == "can_bang_anh":
            preprocess.equalize_hist(conn, id, task_param)
        cursor.close()

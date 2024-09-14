import ftplib
import os.path
from utils.config import *
from utils.preprocessing_image import Preprocessing_Image
import psycopg2
import json
import ast
from datetime import datetime
import threading
import time
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

ftp_directory = json.load(open("ftp_directory.json"))
FTP_ENHANCE_IMAGE_PATH = ftp_directory['enhance_image_result_directory']
FTP_PREPROCESS_IMAGE_PATH = ftp_directory['preprocess_image_result_directory']
FTP_MERGE_IR_PATH = ftp_directory['merge_ir_result_directory']
FTP_MERGE_CHANNEL_PATH = ftp_directory['merge_channel_result_directory']
FTP_FORMAT_CONVERT_PATH = ftp_directory['format_convert_result_directory']
FTP_SHARPEN_IMAGE_PATH = ftp_directory['sharpen_image_result_directory']
FTP_ADJUST_IMAGE_PATH = ftp_directory['adjust_image_result_directory']
FTP_EQUALIZE_IMAGE_PATH = ftp_directory['equalize_image_result_directory']
FTP_ILLUM_CORRECT_IMAGE_PATH = ftp_directory['illum_correct_result_directory']


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
            pass
            # print(f"Error changing to directory '{directory}': {e}")


def download_file(ftp, ftp_file_path, local_file_path):
    try:
        with open(local_file_path, 'wb') as file:
            ftp.retrbinary(f"RETR {ftp_file_path}", file.write)
        # print(f"Downloaded '{ftp_file_path}' to '{local_file_path}'")
    except ftplib.all_errors as e:
        pass
        # print(f"FTP error: {e}")


def route_to_db(cursor):
    cursor.execute('SET search_path TO public')
    cursor.execute("SELECT current_schema()")


def update_database(id, task_stat_value, conn):
    cursor = conn.cursor()
    # Update the task_stat field
    cursor.execute('UPDATE avt_task SET task_stat = %s WHERE id = %s', (task_stat_value, id))
    conn.commit()
    # Select and print the updated row
    # cursor.execute('SELECT * FROM avt_task WHERE id = %s', (id,))
    # row = cursor.fetchone()
    # print(row)


def check_and_update(id, task_stat_value_holder, conn, stop_event):
    start_time = time.time()
    while not stop_event.is_set():
        time.sleep(5)
        if stop_event.is_set():
            break
        elapsed_time = time.time() - start_time
        task_stat_value_holder['value'] = max(2, int(elapsed_time))
        update_database(id, task_stat_value_holder['value'], conn)


def check_epsg_code(tiff_path):
    with rasterio.open(tiff_path) as src:
        crs = src.crs
        if crs:
            epsg_code = int(crs.to_epsg())
            return epsg_code
        else:
            return 0


def convert_epsg_4326(input_tiff_path, output_tiff_path, dst_crs='EPSG:4326'):
    with rasterio.open(input_tiff_path) as src:
        src_profile = src.profile

        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        dst_profile = src_profile.copy()
        dst_profile.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rasterio.open(output_tiff_path, 'w', **dst_profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


def get_time():
    now = datetime.now()
    current_datetime = datetime(now.year, now.month, now.day, now.hour, now.minute, now.second, now.microsecond)
    return current_datetime


def get_time_string():
    now = datetime.now()
    current_datetime = (str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_"
                        + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second))
    return current_datetime


class Preprocessing:
    def __init__(self):
        pass

    def enhance_image(self, conn, id, task_param, ftp):
        input_file = task_param['input_file']
        do_tuong_phan = float(task_param['do_tuong_phan'])
        do_sang = int(task_param['do_sang'])
        do_net = int(task_param['do_net'])
        try:
            filename = input_file.split("/")[-1]
            local_file_path = os.path.join(LOCAL_SRC_ENHANCE_IMAGE_PATH, filename)
            if not os.path.isfile(local_file_path):
                download_file(ftp, input_file, local_file_path)
            epsg_code = check_epsg_code(local_file_path)
            if epsg_code == 0:
                cursor = conn.cursor()
                route_to_db(cursor)
                cursor.execute("UPDATE avt_task SET task_stat = 0 AND task_message = 'EPSG ERROR' WHERE id = %s", (id,))
                conn.commit()
                return False
            elif epsg_code != 4326:
                converted_input_files_local = local_file_path.split(".")[0] + "_4326.tif"
                convert_epsg_4326(local_file_path, converted_input_files_local)
                local_file_path = converted_input_files_local
            date_create = get_time_string()
            output_image_name = "result_enhance_" + format(date_create) + ".tif"
            output_path = os.path.join(LOCAL_RESULT_ENHANCE_IMAGE_PATH, output_image_name)
            preprocess_image = Preprocessing_Image()
            preprocess_image.enhance_image(local_file_path, output_path, do_tuong_phan, do_sang, do_net)
            result_image_name = output_path.split("/")[-1]
            ftp_dir = FTP_ENHANCE_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": [save_dir]
            }).replace("'", "\"")
            with open(output_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
            # owner_group = 'avtadmin:avtadmin'
            # chown_command = f'SITE CHOWN {owner_group} {save_dir}'
            # ftp.sendcmd(chown_command)
            # print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                           (task_output, get_time(), id,))
            conn.commit()
            return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    def check_and_merge_ir(self, conn, id, task_param, ftp):
        input_file = task_param['input_file']
        input_file_ir = task_param['input_file_ir']
        try:
            filename = input_file.split("/")[-1]
            local_file_path = os.path.join(LOCAL_SRC_MERGE_IR_PATH, filename)
            if not os.path.isfile(local_file_path):
                download_file(ftp, input_file, local_file_path)
            epsg_code = check_epsg_code(local_file_path)
            if epsg_code == 0:
                cursor = conn.cursor()
                route_to_db(cursor)
                cursor.execute("UPDATE avt_task SET task_stat = 0 AND task_message = 'EPSG ERROR' WHERE id = %s", (id,))
                conn.commit()
                return False
            elif epsg_code != 4326:
                converted_input_files_local = local_file_path.split(".")[0] + "_4326.tif"
                convert_epsg_4326(local_file_path, converted_input_files_local)
                local_file_path = converted_input_files_local
            date_create = get_time_string()
            output_image_name = "result_merge_ir_" + format(date_create) + ".tif"
            output_path = os.path.join(LOCAL_RESULT_MERGE_IR_PATH, output_image_name)
            preprocess_image = Preprocessing_Image()
            if input_file_ir == "":
                channel_check = preprocess_image.preprocess_no_ir(local_file_path, output_path)
                if not channel_check:
                    cursor = conn.cursor()
                    route_to_db(cursor)
                    cursor.execute(
                        "UPDATE avt_task SET task_stat = 0 AND task_output = 'Can them anh IR' WHERE id = %s", (id,))
                    conn.commit()
                    return False
                else:
                    cursor = conn.cursor()
                    route_to_db(cursor)
                    cursor.execute(
                        "UPDATE avt_task SET task_stat = 0 AND task_output = 'Anh du kenh pho' WHERE id = %s",
                        (id,))
                    conn.commit()
                    return True
            else:
                preprocess_image.preprocess_ir(local_file_path, input_file_ir, output_path)
                ftp_dir = FTP_MERGE_IR_PATH
                ftp.cwd(str(ftp_dir))
                save_dir = ftp_dir + "/" + output_image_name
                task_output = str({
                    "output_image": [save_dir]
                }).replace("'", "\"")
                with open(output_path, "rb") as file:
                    ftp.storbinary(f"STOR {save_dir}", file)
                ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
                owner_group = 'avtadmin:avtadmin'
                chown_command = f'SITE CHOWN {owner_group} {save_dir}'
                ftp.sendcmd(chown_command)
                # print("Connection closed")
                cursor = conn.cursor()
                route_to_db(cursor)
                cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                               (task_output, get_time(), id,))
                conn.commit()
                return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    def merge_channel(self, conn, id, task_param, ftp):
        input_file = task_param['input_file']
        selected_channels = ast.literal_eval(task_param['selected_channels'])
        try:
            filename = input_file.split("/")[-1]
            local_file_path = os.path.join(LOCAL_SRC_MERGE_CHANNEL_PATH, filename)
            if not os.path.isfile(local_file_path):
                download_file(ftp, input_file, local_file_path)
            epsg_code = check_epsg_code(local_file_path)
            if epsg_code == 0:
                cursor = conn.cursor()
                route_to_db(cursor)
                cursor.execute("UPDATE avt_task SET task_stat = 0 AND task_message = 'EPSG ERROR' WHERE id = %s", (id,))
                conn.commit()
                return False
            elif epsg_code != 4326:
                converted_input_files_local = local_file_path.split(".")[0] + "_4326.tif"
                convert_epsg_4326(local_file_path, converted_input_files_local)
                local_file_path = converted_input_files_local
            date_create = get_time_string()
            output_image_name = "result_merge_channel_" + format(date_create) + ".tif"
            output_path = os.path.join(LOCAL_RESULT_MERGE_CHANNEL_PATH, output_image_name)
            preprocess_image = Preprocessing_Image()
            channel_check = preprocess_image.band_check(local_file_path)
            print(channel_check)
            if not channel_check:
                cursor = conn.cursor()
                route_to_db(cursor)
                cursor.execute(
                    "UPDATE avt_task SET task_stat = 0 AND task_message = 'Chua du kenh pho' WHERE id = %s", (id,))
                conn.commit()
                return False
            else:
                preprocess_image.merge_channel(local_file_path, output_path, selected_channels)
                ftp_dir = FTP_MERGE_CHANNEL_PATH
                ftp.cwd(str(ftp_dir))
                save_dir = ftp_dir + "/" + output_image_name
                task_output = str({
                    "output_image": [save_dir]
                }).replace("'", "\"")
                with open(output_path, "rb") as file:
                    ftp.storbinary(f"STOR {save_dir}", file)
                ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
                # owner_group = 'avtadmin:avtadmin'
                # chown_command = f'SITE CHOWN {owner_group} {save_dir}'
                # ftp.sendcmd(chown_command)
                # print("Connection closed")
                cursor = conn.cursor()
                route_to_db(cursor)
                cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                               (task_output, get_time(), id,))
                conn.commit()
                return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    def image_format_convert(self, conn, id, task_param, ftp):
        input_file = task_param['input_file']
        single_bands = task_param['single_bands']
        single_bands = ast.literal_eval(single_bands)
        multi_bands = task_param['multi_bands']
        multi_bands = ast.literal_eval(multi_bands)
        try:
            filename = input_file.split("/")[-1]
            local_file_path = LOCAL_SRC_FORMAT_CONVERT_PATH + filename
            if not os.path.isfile(local_file_path):
                download_file(ftp, input_file, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.image_format_convert(local_file_path, single_bands, multi_bands)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_FORMAT_CONVERT_PATH + "/" + result_image_name.split(".")[0]
            check_and_create_directory(ftp, ftp_dir)
            ftp.cwd(str(ftp_dir))
            export_types = ["png", "jpg", "tiff"]
            task_output_arr = []
            for export_type in export_types:
                filename = result_image_name + "." + export_type
                with open(result_image_path + "." + export_type, "rb") as file:
                    save_dir = ftp_dir + "/" + filename
                    task_output_arr.append(save_dir)
                    ftp.storbinary(f"STOR {save_dir}", file)
                    ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
                    owner_group = 'avtadmin:avtadmin'
                    chown_command = f'SITE CHOWN {owner_group} {save_dir}'
                    ftp.sendcmd(chown_command)
            ftp.sendcmd(f'SITE CHMOD 775 {ftp_dir}')
            # owner_group = 'avtadmin:avtadmin'
            # chown_command = f'SITE CHOWN {owner_group} {ftp_dir}'
            # ftp.sendcmd(chown_command)
            # print("Connection closed")
            task_output = str({
                "png_image_output": task_output_arr[0],
                "jpg_image_output": task_output_arr[1],
                "tiff_image_output": task_output_arr[2]
            }).replace("'", "\"")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                           (task_output, get_time(), id,))
            conn.commit()
            return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    # def image_format_convert(self, conn, id, task_param, ftp):
    #     input_file = task_param['input_file']
    #     try:
    #         filename = input_file.split("/")[-1]
    #         local_file_path = LOCAL_SRC_FORMAT_CONVERT_PATH + filename
    #         if not os.path.isfile(local_file_path):
    #             download_file(ftp, input_file, local_file_path)
    #         preprocess_image = Preprocessing_Image()
    #         result_image_path = preprocess_image.format_convert(local_file_path)
    #         result_image_name = result_image_path.split("/")[-1]
    #         ftp_dir = FTP_FORMAT_CONVERT_PATH + "/" + result_image_name.split(".")[0]
    #         check_and_create_directory(ftp, ftp_dir)
    #         ftp.cwd(str(ftp_dir))
    #         export_types = ["png", "jpg"]
    #         task_output_arr = []
    #         for export_type in export_types:
    #             filename = result_image_name + "." + export_type
    #             with open(result_image_path + "." + export_type, "rb") as file:
    #                 save_dir = ftp_dir + "/" + filename
    #                 task_output_arr.append(save_dir)
    #                 ftp.storbinary(f"STOR {save_dir}", file)
    #                 ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
    #         ftp.sendcmd(f'SITE CHMOD 775 {ftp_dir}')
    #         # print("Connection closed")
    #         task_output = str({
    #             "png_image_output": task_output_arr[0],
    #             "jpg_image_output": task_output_arr[1],
    #         }).replace("'", "\"")
    #         cursor = conn.cursor()
    #         route_to_db(cursor)
    #         cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
    #                        (task_output, get_time(), id,))
    #         conn.commit()
    #         return True
    #     except ftplib.all_errors as e:
    #         cursor = conn.cursor()
    #         route_to_db(cursor)
    #         cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
    #         conn.commit()
    #         # print(f"FTP error: {e}")
    #         return False


    def sharpen_image(self, conn, id, task_param, ftp):
        ORG_input_file = task_param['input_file']
        PAN_input_file = task_param['input_file_single_band']
        try:
            org_filename = ORG_input_file.split("/")[-1]
            local_org_file_path = LOCAL_SRC_SHARPEN_IMAGE_PATH + org_filename
            pan_filename = PAN_input_file.split("/")[-1]
            local_pan_file_path = LOCAL_SRC_SHARPEN_IMAGE_PATH + pan_filename
            if not os.path.isfile(local_org_file_path):
                download_file(ftp, ORG_input_file, local_org_file_path)
            if not os.path.isfile(local_pan_file_path):
                download_file(ftp, PAN_input_file, local_pan_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.sharpen_image(local_org_file_path, local_pan_file_path)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_SHARPEN_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": [save_dir]
            }).replace("'", "\"")
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
            # owner_group = 'avtadmin:avtadmin'
            # chown_command = f'SITE CHOWN {owner_group} {save_dir}'
            # ftp.sendcmd(chown_command)
            # print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                           (task_output, get_time(), id,))
            conn.commit()
            return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    def adjust_gamma(self, conn, id, task_param, ftp):
        src_img_path = task_param['input_file']
        gamma = task_param['gamma']
        try:
            filename = src_img_path.split("/")[-1]
            local_file_path = LOCAL_SRC_ADJUST_IMAGE_PATH + filename
            if not os.path.isfile(local_file_path):
                download_file(ftp, src_img_path, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.adjust_gamma(local_file_path, gamma)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_ADJUST_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": [save_dir]
            }).replace("'", "\"")
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
            # owner_group = 'avtadmin:avtadmin'
            # chown_command = f'SITE CHOWN {owner_group} {save_dir}'
            # ftp.sendcmd(chown_command)
            # print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                           (task_output, get_time(), id,))
            conn.commit()
            return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    def equalize_hist(self, conn, id, task_param, ftp):
        src_img_path = task_param['input_file']
        mode = task_param['mode']
        tileGridSize = task_param['tileGridSize']
        try:
            filename = src_img_path.split("/")[-1]
            local_file_path = LOCAL_SRC_EQUALIZE_IMAGE_PATH + filename
            if not os.path.isfile(local_file_path):
                download_file(ftp, src_img_path, local_file_path)
            preprocess_image = Preprocessing_Image()
            result_image_path = preprocess_image.hist_equalize(local_file_path, mode, tileGridSize)
            result_image_name = result_image_path.split("/")[-1]
            ftp_dir = FTP_EQUALIZE_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": [save_dir]
            }).replace("'", "\"")
            with open(result_image_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
            # owner_group = 'avtadmin:avtadmin'
            # chown_command = f'SITE CHOWN {owner_group} {save_dir}'
            # ftp.sendcmd(chown_command)
            # print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                           (task_output, get_time(), id,))
            conn.commit()
            return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    def illumination_correct(self, conn, id, task_param, ftp):
        input_file = task_param['input_file']
        try:
            filename = input_file.split("/")[-1]
            local_file_path = os.path.join(LOCAL_SRC_ILLUM_CORRECT_IMAGE_PATH, filename)
            if not os.path.isfile(local_file_path):
                download_file(ftp, input_file, local_file_path)
            epsg_code = check_epsg_code(local_file_path)
            if epsg_code == 0:
                cursor = conn.cursor()
                route_to_db(cursor)
                cursor.execute("UPDATE avt_task SET task_stat = 0 AND task_message = 'EPSG ERROR' WHERE id = %s", (id,))
                conn.commit()
                return False
            elif epsg_code != 4326:
                converted_input_files_local = local_file_path.split(".")[0] + "_4326.tif"
                convert_epsg_4326(local_file_path, converted_input_files_local)
                local_file_path = converted_input_files_local
            date_create = get_time_string()
            output_image_name = "result_illum_correct_" + format(date_create) + ".tif"
            output_path = os.path.join(LOCAL_RESULT_ILLUM_CORRECT_IMAGE_PATH, output_image_name)
            preprocess_image = Preprocessing_Image()
            preprocess_image.illumination_correct(local_file_path, output_path)
            result_image_name = output_path.split("/")[-1]
            ftp_dir = FTP_ILLUM_CORRECT_IMAGE_PATH
            ftp.cwd(str(ftp_dir))
            save_dir = ftp_dir + "/" + result_image_name
            task_output = str({
                "output_image": [save_dir]
            }).replace("'", "\"")
            with open(output_path, "rb") as file:
                ftp.storbinary(f"STOR {save_dir}", file)
            ftp.sendcmd(f'SITE CHMOD 775 {save_dir}')
            # owner_group = 'avtadmin:avtadmin'
            # chown_command = f'SITE CHOWN {owner_group} {save_dir}'
            # ftp.sendcmd(chown_command)
            # print("Connection closed")
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 1, task_output = %s, updated_at = %s WHERE id = %s",
                           (task_output, get_time(), id,))
            conn.commit()
            return True
        except ftplib.all_errors as e:
            cursor = conn.cursor()
            route_to_db(cursor)
            cursor.execute("UPDATE avt_task SET task_stat = 0 WHERE id = %s", (id,))
            conn.commit()
            # print(f"FTP error: {e}")
            return False

    def process(self, id, config_data):
        conn = psycopg2.connect(
            dbname=config_data['database']['database'],
            user=config_data['database']['user'],
            password=config_data['database']['password'],
            host=config_data['database']['host'],
            port=config_data['database']['port']
        )
        task_stat_value_holder = {'value': 2}
        stop_event = threading.Event()
        checker_thread = threading.Thread(target=check_and_update, args=(id, task_stat_value_holder, conn, stop_event))
        checker_thread.start()
        try:
            cursor = conn.cursor()
            cursor.execute('SET search_path TO public')
            cursor.execute("SELECT current_schema()")
            cursor.execute("SELECT task_param FROM avt_task WHERE id = %s", (id,))
            result = cursor.fetchone()
            preprocess = Preprocessing()
            task_param = json.loads(result[0])
            algorithm = task_param["algorithm"]
            return_flag = False
            ftp = connect_ftp(config_data)
            if algorithm == "ket_hop_kenh":
                return_flag = preprocess.merge_channel(conn, id, task_param, ftp)
            elif algorithm == "kiem_tra_kenh_ir":
                return_flag = preprocess.check_and_merge_ir(conn, id, task_param, ftp)
            elif algorithm == "lam_sac_net":
                return_flag = preprocess.sharpen_image(conn, id, task_param, ftp)
            elif algorithm == "dieu_chinh_anh":
                return_flag = preprocess.adjust_gamma(conn, id, task_param, ftp)
            elif algorithm == "can_bang_anh":
                return_flag = preprocess.equalize_hist(conn, id, task_param, ftp)
            elif algorithm == "hieu_chinh_sang":
                return_flag = preprocess.illumination_correct(conn, id, task_param, ftp)
            elif algorithm == "ket_xuat_dinh_dang":
                return_flag = preprocess.image_format_convert(conn, id, task_param, ftp)
            elif algorithm == "nang_cao_chat_luong":
                return_flag = preprocess.enhance_image(conn, id, task_param, ftp)
            cursor.close()
            if return_flag:
                task_stat_value_holder['value'] = 1
            else:
                task_stat_value_holder['value'] = 0
        except Exception as e:
            task_stat_value_holder['value'] = 0
        stop_event.set()
        update_database(id, task_stat_value_holder['value'], conn)
        checker_thread.join()

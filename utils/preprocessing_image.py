import numpy as np
import tifffile as tiff
import cv2
import datetime
from datetime import datetime
from utils.convert_to_tiff import convert_to_tiff
from utils.config import *
import rasterio


def normalize_band(band):
    """Normalize the band to the range 0-255."""
    band_min, band_max = band.min(), band.max()
    normalized = (band - band_min) / (band_max - band_min) * 255
    return normalized.astype(np.uint8)


def get_time_string():
    now = datetime.now()
    current_datetime = (str(now.year) + "_" + str(now.month) + "_" + str(now.day) + "_"
                        + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second))
    return current_datetime


class Preprocessing_Image:
    def __init__(self):
        pass

    def enhance_image(self, input_tiff, output_tiff, ALPHA, BETA, KERNEL):
        SHARPEN_KERNEL = np.array([[-1, -1, -1],
                                   [-1, KERNEL, -1],
                                   [-1, -1, -1]])
        with rasterio.open(input_tiff) as src:
            profile = src.profile
            image_data = src.read()
            processed_data = np.zeros_like(image_data)
            for i in range(image_data.shape[0]):
                sharpened_image = cv2.filter2D(image_data[i], -1, SHARPEN_KERNEL)
                contrast_image = cv2.convertScaleAbs(sharpened_image, alpha=ALPHA, beta=BETA)
                processed_data[i] = cv2.equalizeHist(contrast_image)

            with rasterio.open(output_tiff, 'w', **profile) as dst:
                dst.write(processed_data)

    def band_check(self, tiff_image_path):
        with rasterio.open(tiff_image_path) as src:
            # Read the channels
            channels = [src.read(i + 1) for i in range(src.count)]
            # print(len(channels))

            # Check if there are 4 channels (including IR)
            ir_channel = channels[-1]
            rgb_channels = channels[:-1]

            # Detect if there are identical channels
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    if np.array_equal(channels[i], channels[j]):
                        return False
        return True

    def preprocess_no_ir(self, tiff_image_path, output_path):
        with rasterio.open(tiff_image_path) as src:
            # Read the channels
            channels = [src.read(i + 1) for i in range(src.count)]
            # print(len(channels))

            # Check if there are 4 channels (including IR)
            ir_channel = channels[-1]
            rgb_channels = channels[:-1]

            # Detect if there are identical channels
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    if np.array_equal(channels[i], channels[j]):
                        return False

            # Combine channels into a 4-channel array (RGB first, IR last)
            combined_image = np.stack(rgb_channels + [ir_channel])

            # Write the combined image to a new TIFF file
            with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=combined_image.shape[1],
                    width=combined_image.shape[2],
                    count=4,  # 4 channels
                    dtype=combined_image.dtype
            ) as dst:
                for i in range(4):
                    dst.write(combined_image[i], i + 1)

        return True

    def preprocess_ir(self, tiff_image_path, tiff_image_ir_path, output_path):
        check_channel = False
        with rasterio.open(tiff_image_path) as src:
            # Read the channels
            channels = [src.read(i + 1) for i in range(src.count)]
            if len(channels) == 3:
                check_channel = True
        if not check_channel:
            with rasterio.open(tiff_image_path) as src_4ch:
                rgb_channels = src_4ch.read([1, 2, 3])  # Read the first 3 channels (RGB)

            # Open the IR image
            with rasterio.open(tiff_image_ir_path) as src_ir:
                ir_channel = src_ir.read(1)  # Assuming IR image has a single channel

            # Stack the RGB channels and the IR channel to create a 4-channel image
            combined_image = np.vstack((rgb_channels, np.expand_dims(ir_channel, axis=0)))

            # Write the combined image to a new 4-channel TIFF file
            with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=combined_image.shape[1],
                    width=combined_image.shape[2],
                    count=4,  # 4 channels
                    dtype=combined_image.dtype
            ) as dst:
                for i in range(4):
                    dst.write(combined_image[i], i + 1)
        else:
            with rasterio.open(tiff_image_path) as src_rgb:
                rgb_channels = src_rgb.read()

            # Open the IR image
            with rasterio.open(tiff_image_ir_path) as src_ir:
                ir_channel = src_ir.read(1)  # Assuming IR image has a single channel

            # Stack the RGB channels and the IR channel to create a 4-channel image
            combined_image = np.vstack((rgb_channels, np.expand_dims(ir_channel, axis=0)))

            # Write the combined image to a new 4-channel TIFF file
            with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=combined_image.shape[1],
                    width=combined_image.shape[2],
                    count=4,  # 4 channels
                    dtype=combined_image.dtype
            ) as dst:
                for i in range(4):
                    dst.write(combined_image[i], i + 1)

    def merge_channel(self, input_tiff, output_tiff, selected_channels):
        # selected_channels = np.array(selected_channels) + 1
        # print(selected_channels)
        with rasterio.open(input_tiff) as src:
            selected_data = src.read(selected_channels)

            output_meta = src.meta.copy()
            output_meta.update({
                'count': len(selected_channels),  # Số kênh đầu ra
                'dtype': selected_data.dtype
            })

            with rasterio.open(output_tiff, 'w', **output_meta) as dst:
                dst.write(selected_data)

    def image_format_convert(self, tiff_image_path, single_bands, multi_bands):
        """

            combine image channel.
            Args:
                tiff_image_path (str): path to tiff image
                single_bands (List[Integer]): list chosen band. e.g: [0, 3, 4]
                multi_bands (List[List[Integer], Integer]): list [multi-bands, combine mode(max: 0 or average: 1)]
                                                            e.g: [[[3, 5, 4], 0], [[4, 2], 1]]
            Returns:
                jpg image
                png image
                tiff image
            Examples:

            """
        input_image = tiff.imread(tiff_image_path)
        bands_list = []
        if len(single_bands) == 0 and len(multi_bands) == 0:  # set default value
            single_bands = [0, 1, 2]
        if len(single_bands) > 0:
            for s_band in single_bands:
                bands_list.append(input_image[s_band, :, :])
        if len(multi_bands) > 0:
            for [m_band, mode] in multi_bands:
                merge_band = []
                if mode == 0:  # max value
                    for band in m_band:
                        merge_band.append(input_image[band, :, :])
                    bands_list.append(np.maximum.reduce(merge_band))
                elif mode == 1:  # average value
                    for band in m_band:
                        merge_band.append(input_image[band, :, :])
                    bands_list.append(np.mean(merge_band, axis=0))
        bands_list_normalize = [normalize_band(band) for band in bands_list]
        bands_stack = np.stack(bands_list_normalize, axis=2)
        date_create = get_time_string()
        tiff_image_name = tiff_image_path.split("/")[-1]
        image_name_output = tiff_image_name.split(".")[0] + "_" + format(date_create)
        result_image_path = LOCAL_RESULT_FORMAT_CONVERT_PATH + "/" + image_name_output
        export_types = ["png", "jpg", "tiff"]
        for image_type in export_types:
            if image_type == "png":
                cv2.imwrite(result_image_path + ".png", bands_stack)
            elif image_type == "jpg":
                cv2.imwrite(result_image_path + ".jpg", bands_stack)
            elif image_type == "tiff":
                image_name_output = result_image_path + ".tif"
                convert_to_tiff(tiff_image_path, image_name_output, bands_stack)
        return result_image_path

    def sharpen_image(self, ORG_image_path, PAN_image_path, contrast=3.0, brightness=20):
        """
            Áp dụng thuật toán Brovey pansharpen trên ảnh đa kênh

            Args:
            ORG_image_path: str
                ảnh độ phân giải thấp đa kênh.
            PAN_image_path: str
                ảnh toàn sắc độ phân giải cao.
            contrast: độ tương phản
            brightness: độ sáng
            Returns:
            numpy.ndarray
                Ảnh pansharpened với độ phân giải cao.
            """
        ORG_image = cv2.imread(ORG_image_path)
        PAN_image = cv2.imread(PAN_image_path)
        PAN_image = cv2.cvtColor(PAN_image, code=cv2.COLOR_BGR2GRAY)
        num_bands = int(ORG_image.shape[2])
        band_weights = [1 / num_bands for _ in range(num_bands)]
        ORG_image = cv2.resize(ORG_image, (PAN_image.shape[1], PAN_image.shape[0]))
        ORG_image = ORG_image.astype(np.float32)
        PAN_image = PAN_image.astype(np.float32)
        channels = [ORG_image[:, :, i] for i in range(ORG_image.shape[2])]
        weighted_sum = sum(w * ch for w, ch in zip(band_weights, channels))
        weighted_sum[weighted_sum == 0] = 1e-6
        ratios = [w * ch / weighted_sum for w, ch in zip(band_weights, channels)]
        sharp_channels = [ratio * PAN_image for ratio in ratios]
        sharp_image = np.stack(sharp_channels, axis=-1)
        sharp_image = np.clip(sharp_image, 0, 255).astype(np.uint8)
        result = cv2.convertScaleAbs(sharp_image, alpha=contrast, beta=brightness)
        image_name = ORG_image_path.split("/")[-1]
        date_create = get_time_string()
        image_name_output = "sharpen_image_" + image_name.split(".")[0] + "_" + format(date_create) + ".tif"
        result_image_path = LOCAL_RESULT_SHARPEN_IMAGE_PATH + "/" + image_name_output
        # cv2.imwrite(result_image_path, result)
        convert_to_tiff(ORG_image_path, result_image_path, result)
        return result_image_path

    def adjust_gamma(self, src_img_path, gamma=0.5):
        src_img = cv2.imread(src_img_path)
        assert 0 < gamma < 1
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        result = cv2.LUT(src_img, lookUpTable)
        image_name = src_img_path.split("/")[-1]
        date_create = get_time_string()
        image_name_output = "adjust_image_" + image_name.split(".")[0] + "_" + format(date_create) + ".tif"
        result_image_path = LOCAL_RESULT_ADJUST_IMAGE_PATH + "/" + image_name_output
        # cv2.imwrite(result_image_path, result)
        convert_to_tiff(src_img_path, result_image_path, result)
        return result_image_path

    def hist_equalize(self, src_img_path, mode="tiles", tileGridSize=8):
        """
        hist_equalize colored image
        """
        src_img = cv2.imread(src_img_path)
        assert mode in ["global", "tiles"]
        ycrcb_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2YCrCb)
        ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])
        if mode == "global":
            ycrcb_img = cv2.equalizeHist(ycrcb_img)
        if mode == "tiles":
            assert type(tileGridSize) is int
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tileGridSize, tileGridSize))
            for i in range(ycrcb_img.shape[-1]):
                ycrcb_img[..., i] = clahe.apply(ycrcb_img[..., i])
        result = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        image_name = src_img_path.split("/")[-1]
        date_create = get_time_string()
        image_name_output = "equalize_image_" + image_name.split(".")[0] + "_" + format(date_create) + ".tif"
        result_image_path = LOCAL_RESULT_EQUALIZE_IMAGE_PATH + "/" + image_name_output
        # cv2.imwrite(result_image_path, result)
        convert_to_tiff(src_img_path, result_image_path, result)
        return result_image_path

    def illumination_correct(self, input_image_path, output_image_path, num_bands=60):
        with rasterio.open(input_image_path) as src:
            image = src.read()
            profile = src.profile

        bands, rows, cols = image.shape
        band_height = rows // num_bands

        mean_brightness_per_band = np.zeros((bands, num_bands))
        for band in range(bands):
            for i in range(num_bands):
                band_section = image[band, i * band_height:(i + 1) * band_height, :]
                mean_brightness = np.mean(band_section)
                mean_brightness_per_band[band, i] = mean_brightness

        corrected_image = np.copy(image)
        for band in range(bands):
            for i in range(num_bands):
                band_section = corrected_image[band, i * band_height:(i + 1) * band_height, :]
                correction_factor = np.mean(mean_brightness_per_band[band]) / mean_brightness_per_band[band, i]
                corrected_image[band, i * band_height:(i + 1) * band_height, :] = np.clip(
                    band_section * correction_factor, 0, 255)

        with rasterio.open(output_image_path, 'w', **profile) as dst:
            dst.write(corrected_image)

    def format_convert(self, tiff_image_path):
        input_image = cv2.imread(tiff_image_path)
        date_create = get_time_string()
        tiff_image_name = tiff_image_path.split("/")[-1]
        image_name_output = tiff_image_name.split(".")[0] + "_" + format(date_create)
        result_image_path = LOCAL_RESULT_FORMAT_CONVERT_PATH + "/" + image_name_output
        export_types = ["png", "jpg"]
        for image_type in export_types:
            if image_type == "png":
                cv2.imwrite(result_image_path + ".png", input_image)
            elif image_type == "jpg":
                cv2.imwrite(result_image_path + ".jpg", input_image)
        return result_image_path

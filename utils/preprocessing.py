import numpy as np
import tifffile as tiff
import cv2
import datetime
from utils.convert_to_tiff import convert_to_tiff
from utils.config import *


def normalize_band(band):
    """Normalize the band to the range 0-255."""
    band_min, band_max = band.min(), band.max()
    normalized = (band - band_min) / (band_max - band_min) * 255
    return normalized.astype(np.uint8)


class Preprocessing:
    def __init__(self):
        pass

    def merge_image(self, tiff_image_path, single_bands, multi_bands):
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
            single_bands = [0, 3, 4]
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
        date_create = str(datetime.datetime.now().date()).replace('-', '_')
        tiff_image_name = tiff_image_path.split("/")[-1]
        image_name_output = tiff_image_name.split(".")[0] + "_" + format(date_create)
        result_image_path = LOCAL_RESULT_MERGE_IMAGE_PATH + image_name_output
        export_types = ["png", "jpg", "tiff"]
        for image_type in export_types:
            if image_type == "png":
                cv2.imwrite(image_name_output + ".png", bands_stack)
            elif image_type == "jpg":
                cv2.imwrite(image_name_output + ".jpg", bands_stack)
            elif image_type == "tiff":
                image_name_output = image_name_output + ".tiff"
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
        date_create = str(datetime.datetime.now().date()).replace('-', '_')
        image_name_output = "sharpen_image_" + image_name.split(".")[0] + "_" + format(date_create) + ".jpg"
        result_image_path = LOCAL_RESULT_SHARPEN_IMAGE_PATH + image_name_output
        cv2.imwrite(image_name_output, result)
        return result_image_path

    def adjust_gamma(self, src_img_path, gamma=0.5):
        src_img = cv2.imread(src_img_path)
        assert 0 < gamma < 1
        lookUpTable = np.empty((1, 256), np.uint8)
        for i in range(256):
            lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

        result = cv2.LUT(src_img, lookUpTable)
        image_name = src_img_path.split("/")[-1]
        date_create = str(datetime.datetime.now().date()).replace('-', '_')
        image_name_output = "adjust_image_" + image_name.split(".")[0] + "_" + format(date_create) + ".jpg"
        result_image_path = LOCAL_RESULT_SHARPEN_IMAGE_PATH + image_name_output
        cv2.imwrite(image_name_output, result)
        return result_image_path

    def hist_equalize(self, src_img_path, mode="tiles", tileGridSize=(8, 8)):
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
            assert type(tileGridSize) is tuple
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=tileGridSize)
            for i in range(ycrcb_img.shape[-1]):
                ycrcb_img[..., i] = clahe.apply(ycrcb_img[..., i])
        result = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        image_name = src_img_path.split("/")[-1]
        date_create = str(datetime.datetime.now().date()).replace('-', '_')
        image_name_output = "equalize_image_" + image_name.split(".")[0] + "_" + format(date_create) + ".jpg"
        result_image_path = LOCAL_RESULT_SHARPEN_IMAGE_PATH + image_name_output
        cv2.imwrite(image_name_output, result)
        return result_image_path

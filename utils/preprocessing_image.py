import cv2
import datetime
from datetime import datetime
from utils.convert_to_tiff import convert_to_tiff
from utils.config import *
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient
import numpy as np
from rasterio.enums import Resampling
from scipy.optimize import least_squares
from skimage.exposure import match_histograms
from osgeo import gdal
from rasterio.crs import CRS
from rasterio.warp import reproject, calculate_default_transform


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


def residuals(params, src_points, dst_points):
    transformed_points = polynomial_transform(params, src_points)
    return (transformed_points - dst_points).ravel()


def polynomial_transform(params, src_points):
    a0, a1, a2, b0, b1, b2 = params
    x = src_points[:, 0]
    y = src_points[:, 1]
    x_new = a0 + a1 * x + a2 * y
    y_new = b0 + b1 * x + b2 * y
    return np.column_stack((x_new, y_new))


def transform_image(img, params):
    a0, a1, a2, b0, b1, b2 = params
    rows, cols = img.shape
    map_x = np.zeros((rows, cols), dtype=np.float32)
    map_y = np.zeros((rows, cols), dtype=np.float32)

    for y in range(rows):
        for x in range(cols):
            new_x = a0 + a1 * x + a2 * y
            new_y = b0 + b1 * x + b2 * y
            map_x[y, x] = new_x
            map_y[y, x] = new_y

    corrected_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return corrected_img


def get_ortho_proj(crs_string):
    return CRS.from_string(crs_string)

class Preprocessing_Image:
    def __init__(self):
        pass

    def enhance_image(self, input_path, output_path, ALPHA, BETA, SHARPEN_KERNEL):
        SHARPEN_KERNEL = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]])
        with rasterio.open(input_path) as src:
            img = src.read([1, 2, 3])
            metadata = src.meta
        img = np.transpose(img, (1, 2, 0))
        img = img.astype(np.uint8)
        img = cv2.convertScaleAbs(img, alpha=ALPHA, beta=BETA)
        img = cv2.filter2D(img, -1, SHARPEN_KERNEL)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img_hsv[:, :, 2] = cv2.equalizeHist(img_hsv[:, :, 2])
        img_result = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        img_result = np.transpose(img_result, (2, 0, 1))
        metadata.update({
            'driver': 'GTiff',
            'height': img_result.shape[1],
            'width': img_result.shape[2],
            'count': img_result.shape[0],  # Số kênh
            'dtype': img_result.dtype
        })
        with rasterio.open(output_path, 'w', **metadata) as dst:
            dst.write(img_result)

    def band_check(self, tiff_image_path):
        with rasterio.open(tiff_image_path) as src:
            channels = [src.read(i + 1) for i in range(src.count)]
            ir_channel = channels[-1]
            rgb_channels = channels[:-1]
            if len(channels) < 4:
                return 0  # Thiếu kênh phổ
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    if np.array_equal(channels[i], channels[j]):
                        if i == 0:
                            return 1  # Có trùng lặp kênh phổ 1
                        elif i == 1:
                            return 2  # Có trùng lặp kênh phổ 2
                        else:
                            return 3  # Có trùng lặp kênh phổ 3
        return 10  # Đủ kênh phổ

    def preprocess_no_ir(self, tiff_image_path, output_path):
        with rasterio.open(tiff_image_path) as src:
            channels = [src.read(i + 1) for i in range(src.count)]
            ir_channel = channels[-1]
            rgb_channels = channels[:-1]
            for i in range(len(channels)):
                for j in range(i + 1, len(channels)):
                    if np.array_equal(channels[i], channels[j]):
                        return False
            combined_image = np.stack(rgb_channels + [ir_channel])
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
            channels = [src.read(i + 1) for i in range(src.count)]
            if len(channels) == 3:
                check_channel = True
        if not check_channel:
            with rasterio.open(tiff_image_path) as src_4ch:
                rgb_channels = src_4ch.read([1, 2, 3])
            with rasterio.open(tiff_image_ir_path) as src_ir:
                ir_channel = src_ir.read(1)
            combined_image = np.vstack((rgb_channels, np.expand_dims(ir_channel, axis=0)))
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
            with rasterio.open(tiff_image_ir_path) as src_ir:
                ir_channel = src_ir.read(1)
            combined_image = np.vstack((rgb_channels, np.expand_dims(ir_channel, axis=0)))
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

    # def image_format_convert(self, tiff_image_path, single_bands, multi_bands):
    #     """
    #
    #         combine image channel.
    #         Args:
    #             tiff_image_path (str): path to tiff image
    #             single_bands (List[Integer]): list chosen band. e.g: [0, 3, 4]
    #             multi_bands (List[List[Integer], Integer]): list [multi-bands, combine mode(max: 0 or average: 1)]
    #                                                         e.g: [[[3, 5, 4], 0], [[4, 2], 1]]
    #         Returns:
    #             jpg image
    #             png image
    #             tiff image
    #         Examples:
    #
    #         """
    #     input_image = tiff.imread(tiff_image_path)
    #     bands_list = []
    #     if len(single_bands) == 0 and len(multi_bands) == 0:  # set default value
    #         single_bands = [0, 1, 2]
    #     if len(single_bands) > 0:
    #         for s_band in single_bands:
    #             bands_list.append(input_image[s_band, :, :])
    #     if len(multi_bands) > 0:
    #         for [m_band, mode] in multi_bands:
    #             merge_band = []
    #             if mode == 0:  # max value
    #                 for band in m_band:
    #                     merge_band.append(input_image[band, :, :])
    #                 bands_list.append(np.maximum.reduce(merge_band))
    #             elif mode == 1:  # average value
    #                 for band in m_band:
    #                     merge_band.append(input_image[band, :, :])
    #                 bands_list.append(np.mean(merge_band, axis=0))
    #     bands_list_normalize = [normalize_band(band) for band in bands_list]
    #     bands_stack = np.stack(bands_list_normalize, axis=2)
    #     date_create = get_time_string()
    #     tiff_image_name = tiff_image_path.split("/")[-1]
    #     image_name_output = tiff_image_name.split(".")[0] + "_" + format(date_create)
    #     result_image_path = LOCAL_RESULT_FORMAT_CONVERT_PATH + "/" + image_name_output
    #     export_types = ["png", "jpg", "tiff"]
    #     for image_type in export_types:
    #         if image_type == "png":
    #             cv2.imwrite(result_image_path + ".png", bands_stack)
    #         elif image_type == "jpg":
    #             cv2.imwrite(result_image_path + ".jpg", bands_stack)
    #         elif image_type == "tiff":
    #             image_name_output = result_image_path + ".tif"
    #             convert_to_tiff(tiff_image_path, image_name_output, bands_stack)
    #     return result_image_path

    def image_format_convert(self, tiff_path, output_path, polygon_coords, selected_channels, new_resolution,
                             output_formats):
        with rasterio.open(tiff_path) as src:
            polygon = Polygon(polygon_coords)
            oriented_polygon = orient(polygon, sign=1.0)
            out_image, out_transform = mask(src, [oriented_polygon], crop=True)
            original_width = out_image.shape[2]
            original_height = out_image.shape[1]
            new_width = int(original_width * new_resolution)
            new_height = int(original_height * new_resolution)

            resized_channels = []
            for i in selected_channels:
                resized_channel = np.empty((new_height, new_width), dtype=out_image.dtype)
                rasterio.warp.reproject(
                    out_image[i - 1],
                    resized_channel,
                    src_transform=out_transform,
                    src_crs=src.crs,
                    dst_transform=out_transform,
                    dst_crs=src.crs,
                    resampling=Resampling.bilinear
                )
                resized_channels.append(resized_channel)

            merged_image = np.stack(resized_channels, axis=0)
            for output_format in output_formats:
                if output_format in ['png', 'jpg']:
                    merged_image_save = np.moveaxis(merged_image, 0, -1)
                    if len(selected_channels) == 4:
                        merged_image_save = cv2.cvtColor(merged_image_save, cv2.COLOR_RGBA2BGR)
                    output_path_save = output_path + "." + output_format
                    if merged_image_save.dtype == np.float32 or merged_image_save.dtype == np.float64:
                        # Chuyển đổi dữ liệu từ float về int nếu cần
                        merged_image_save = (merged_image_save * 255).astype(np.uint8)
                    cv2.imwrite(output_path_save, merged_image_save)

                elif output_format in ['8_bit']:
                    output_path_save = output_path + "_8_bit.tif"
                    merged_image_8bit = (merged_image / merged_image.max() * 255).astype(np.uint8)
                    out_meta = src.meta.copy()
                    out_meta.update({"dtype": 'uint8', "driver": "GTiff"})
                    with rasterio.open(output_path_save, "w", **out_meta) as dest:
                        dest.write(merged_image_8bit)

                elif output_format in ['16_bit']:
                    output_path_save = output_path + "_16_bit.tif"
                    merged_image_16bit = (merged_image / merged_image.max() * 65535).astype(np.uint16)
                    out_meta = src.meta.copy()
                    out_meta.update({"dtype": 'uint16', "driver": "GTiff"})
                    with rasterio.open(output_path_save, "w", **out_meta) as dest:
                        dest.write(merged_image_16bit)
                else:
                    output_path_save = output_path + ".tif"
                    out_meta = src.meta.copy()
                    out_meta.update({
                        "driver": "GTiff",
                        "height": new_height,
                        "width": new_width,
                        "transform": out_transform,
                        "count": len(selected_channels)
                    })
                    with rasterio.open(output_path_save, "w", **out_meta) as dest:
                        dest.write(merged_image)

    def sharpen_image(self, ORG_image_path, PAN_image_path):
        weights = np.array([0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125])

        with rasterio.open(ORG_image_path) as pan_dataset:
            HR_PAN = pan_dataset.read(1).astype(np.float32)
            pan_meta = pan_dataset.meta

        with rasterio.open(PAN_image_path) as lr_dataset:
            LR_8CH = lr_dataset.read().astype(np.float32)
            lr_meta = lr_dataset.meta

        height, width = HR_PAN.shape
        LR_8CH_resized = np.zeros((LR_8CH.shape[0], height, width), dtype=np.float32)

        for i in range(LR_8CH.shape[0]):
            LR_8CH_resized[i, :, :] = np.resize(LR_8CH[i, :, :], (height, width))

        channels = [LR_8CH_resized[i, :, :] for i in range(LR_8CH_resized.shape[0])]
        HR_PAN = channels[0]
        weighted_sum = sum(w * ch for w, ch in zip(weights, channels))
        weighted_sum[weighted_sum == 0] = 1e-6
        ratios = [w * ch / weighted_sum for w, ch in zip(weights, channels)]
        sharp_channels = [ratio * HR_PAN for ratio in ratios]
        sharp_image = np.stack(sharp_channels, axis=0)
        pan_meta.update({
            "count": LR_8CH.shape[0],
            "dtype": "uint8"
        })
        image_name = ORG_image_path.split("/")[-1]
        date_create = get_time_string()
        image_name_output = "sharpen_image_" + image_name.split(".")[0] + "_" + format(date_create) + ".tif"
        output_path = LOCAL_RESULT_SHARPEN_IMAGE_PATH + "/" + image_name_output
        with rasterio.open(output_path, 'w', **pan_meta) as dst:
            dst.write(np.clip(sharp_image, 0, 255).astype(np.uint8))
        return output_path

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

    def physical_error_correction(self, input_file, output_file, distortion_factor):
        assert float(cv2.__version__.rsplit('.', 1)[0]) >= 3, 'OpenCV version 3 or newer required.'

        # Focal lengths fx and fy remain unchanged
        fx = 5000
        fy = 5000

        with rasterio.open(input_file) as src:
            width = src.width
            height = src.height

            # Calculate cx and cy based on the image dimensions
            cx = (width - 1) / 2
            cy = (height - 1) / 2

            # Construct the new intrinsic matrix K
            K = np.array([[fx, 0., cx],
                          [0., fy, cy],
                          [0., 0., 1]])

            # Read the first three channels (RGB)
            rgb_image = src.read([1, 2, 3]).transpose(1, 2, 0)

            # Read the IR channel
            ir_channel = src.read(4)

        # Zero distortion coefficients
        D = np.array([0., 0., 0., 0.])

        # Use Knew to scale the output
        Knew = K.copy()
        Knew[(0, 1), (0, 1)] = distortion_factor * Knew[(0, 1), (0, 1)]

        # Apply fisheye undistortion to the RGB channels
        img_undistorted_rgb = cv2.fisheye.undistortImage(rgb_image, K, D=D, Knew=Knew)

        # Stack the undistorted RGB and the IR channel back together
        img_undistorted = np.dstack((img_undistorted_rgb, ir_channel))

        # Define the target CRS (EPSG:4326)
        dst_crs = 'EPSG:4326'

        # Define the transform and reproject settings
        outfile_temp = output_file[0:-4] + '_temp.tif'
        with rasterio.open(
                outfile_temp,
                'w',
                driver='GTiff',
                height=img_undistorted.shape[0],
                width=img_undistorted.shape[1],
                count=4,  # Four channels: R, G, B, IR
                dtype=img_undistorted.dtype,
                crs=src.crs,  # Using the source CRS initially
                transform=src.transform
        ) as dst:
            # Write each channel to the TIFF
            dst.write(img_undistorted[:, :, 0], 1)  # Red channel
            dst.write(img_undistorted[:, :, 1], 2)  # Green channel
            dst.write(img_undistorted[:, :, 2], 3)  # Blue channel
            dst.write(img_undistorted[:, :, 3], 4)  # IR channel

            # Now reproject the image to EPSG:4326
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, dst.width, dst.height, *dst.bounds)

            with rasterio.open(
                    output_file,
                    'w',
                    driver='GTiff',
                    height=height,
                    width=width,
                    count=4,
                    dtype=img_undistorted.dtype,
                    crs=dst_crs,  # Set to EPSG:4326
                    transform=transform
            ) as dst_reprojected:
                for i in range(1, 5):  # Loop through each channel
                    reproject(
                        source=rasterio.band(dst, i),
                        destination=rasterio.band(dst_reprojected, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )

    def radiometric_correction(self, input_path, output_path, png_paths):
        with rasterio.open(input_path) as tiff_src:
            tiff_img = tiff_src.read()
            profile = tiff_src.profile
        rgb_channels = tiff_img[:3]
        ir_channel = tiff_img[3]
        png_histograms = []
        for png_path in png_paths:
            png_img = cv2.imread(png_path)
            png_img_rgb = cv2.cvtColor(png_img, cv2.COLOR_BGR2RGB)
            png_histograms.append(png_img_rgb)
        png_histograms = np.array(png_histograms)
        median_histogram = np.median(png_histograms, axis=0).astype(np.uint8)
        adjusted_rgb = np.zeros_like(rgb_channels, dtype=np.uint8)
        for channel in range(3):
            adjusted_rgb[channel] = match_histograms(rgb_channels[channel], median_histogram[:, :, channel])
        adjusted_tiff = np.vstack((adjusted_rgb, np.expand_dims(ir_channel, axis=0)))
        with rasterio.open(output_path, 'w', **profile) as tiff_dst:
            tiff_dst.write(adjusted_tiff)

    def geometric_correction(self, input_path, output_path, src_points, dst_points):
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        initial_guess = np.zeros(6)
        result = least_squares(residuals, initial_guess, args=(src_points, dst_points))
        params = result.x
        with rasterio.open(input_path) as src:
            num_channels = src.count
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            channels = []
            for i in range(1, num_channels + 1):
                channel_data = src.read(i)
                corrected_channel = transform_image(channel_data, params)
                channels.append(corrected_channel)
        corrected_image = np.array(channels)
        new_metadata = {
            'driver': 'GTiff',
            'count': num_channels,  #
            'height': height,
            'width': width,
            'dtype': corrected_image.dtype,
            'crs': crs,
            'transform': transform
        }

        # Bước 3: Lưu ảnh TIFF với metadata mới
        with rasterio.open(output_path, 'w', **new_metadata) as dst:
            dst.write(corrected_image)

    def gcp_correction(self, input_path, output_path, src_points, dst_points):
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)
        initial_guess = np.zeros(6)
        result = least_squares(residuals, initial_guess, args=(src_points, dst_points))
        params = result.x
        with rasterio.open(input_path) as src:
            num_channels = src.count
            height, width = src.height, src.width
            crs = src.crs
            transform = src.transform
            channels = []
            for i in range(1, num_channels + 1):
                channel_data = src.read(i)
                corrected_channel = transform_image(channel_data, params)
                channels.append(corrected_channel)
        corrected_image = np.array(channels)
        new_metadata = {
            'driver': 'GTiff',
            'count': num_channels,  #
            'height': height,
            'width': width,
            'dtype': corrected_image.dtype,
            'crs': crs,
            'transform': transform
        }
        with rasterio.open(output_path, 'w', **new_metadata) as dst:
            dst.write(corrected_image)

    def dem_correction(self, aerial_image_path, dem_path, output_path, lon_angle, lat_angle):
        with rasterio.open(aerial_image_path) as source:
            src_crs = "EPSG:4326"  # Hệ tọa độ của RPCs (ảnh nguồn có CRS là EPSG:4326)

            # Lấy kích thước từ ảnh gốc
            src_width = source.width
            src_height = source.height
            dst_crs = "EPSG:4326"  # Đảm bảo đầu ra có CRS là EPSG:4326

            # Optional keyword arguments to be passed to GDAL transformer
            kwargs = {
                'RPC_DEM': dem_path
            }

            # Kiểm tra số lượng kênh (bands) của ảnh nguồn
            num_bands = source.count

            # Tạo mảng đầu ra cho ảnh đích với kích thước giống với ảnh gốc
            destination = np.zeros((num_bands, src_height, src_width), dtype=np.uint8)

            # Xử lý từng kênh của ảnh
            for i in range(1, num_bands + 1):  # Duyệt qua các kênh từ 1 đến num_bands
                _, dst_transform = reproject(
                    rasterio.band(source, i),
                    destination[i - 1],  # Chọn đúng kênh trong destination
                    rpcs=source.rpcs,
                    src_crs=src_crs,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                    **kwargs
                )

            # Lưu kết quả ra file TIFF mới với hệ tọa độ EPSG:4326 và kích thước giống ảnh gốc
            new_tiff_path = output_path
            with rasterio.open(
                    new_tiff_path, 'w',
                    driver='GTiff',
                    height=src_height,  # Chiều cao lấy từ ảnh gốc
                    width=src_width,  # Chiều rộng lấy từ ảnh gốc
                    count=num_bands,  # Số lượng kênh giống ảnh gốc
                    dtype=destination.dtype,
                    crs=dst_crs,  # Đảm bảo CRS là EPSG:4326
                    transform=dst_transform
            ) as dst:
                # Ghi tất cả các kênh vào file
                for i in range(1, num_bands + 1):
                    dst.write(destination[i - 1], i)  # Ghi từng kênh vào band tương ứng

    def dem_band_check(self, dem_path):
        with rasterio.open(dem_path) as src:
            num_channels = src.count
        if num_channels == 1:
            return True
        else:
            return False

    def orthogonal_correct(self, input_file, output_file, crs_string):
        dst_crs = get_ortho_proj(crs_string=crs_string)

        # Mở file ảnh gốc 4 kênh (RGB và IR)
        dataset = rasterio.open(input_file)
        orig_data = dataset.read()  # Đọc tất cả các kênh, kết quả là mảng (count, height, width)

        # Tính toán transform và kích thước của ảnh khi reproject
        transform, width, height = rasterio.warp.calculate_default_transform(
            dataset.crs, dst_crs, dataset.width, dataset.height,
            left=dataset.bounds.left, right=dataset.bounds.right,
            top=dataset.bounds.top, bottom=dataset.bounds.bottom
        )

        # Khởi tạo mảng ortho_data với kích thước tương tự và kiểu dữ liệu phù hợp
        ortho_data = np.zeros((dataset.count, height, width), dtype=orig_data.dtype)

        # Reproject từng kênh của ảnh
        for i in range(1, dataset.count + 1):
            reproject(
                source=orig_data[i - 1],  # Truy cập kênh thứ i (index 0-based)
                destination=ortho_data[i - 1],  # Ghi kênh tương ứng
                src_transform=dataset.transform,
                src_crs=dataset.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear,
                dst_nodata=0
            )

        # Lưu ortho_data thành tệp TIFF với đủ 4 kênh
        with rasterio.open(
                output_file, 'w',
                driver='GTiff',
                height=height, width=width,
                count=dataset.count,  # Ghi đủ số kênh
                dtype=ortho_data.dtype,
                crs=dst_crs, transform=transform
        ) as dst:
            dst.write(ortho_data)

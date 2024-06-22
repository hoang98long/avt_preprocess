from osgeo import gdal, gdal_array
import cv2


def read_tiff_image(tiff_path):
    # Open the GeoTIFF file
    dataset = gdal.Open(tiff_path, gdal.GA_ReadOnly)

    if dataset is None:
        print("Failed to open the GeoTIFF file.")
        return None

    # Read the image data
    image_data = dataset.ReadAsArray()

    # Get the number of bands, width, and height of the image
    num_bands = dataset.RasterCount
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # Close the dataset
    dataset = None

    return image_data, num_bands, width, height


def save_tiff_image(image_data, output_tiff_path, original_tiff_path):
    # Open the original GeoTIFF file to get the metadata
    original_dataset = gdal.Open(original_tiff_path, gdal.GA_ReadOnly)

    if original_dataset is None:
        print("Failed to open the original GeoTIFF file.")
        return

    # Create a new GeoTIFF file
    driver = gdal.GetDriverByName("GTiff")

    # Define the dataset dimensions (width, height)
    height, width = image_data.shape[1:]

    # Create the output dataset
    output_dataset = driver.Create(output_tiff_path, width, height, 3,
                                   gdal_array.NumericTypeCodeToGDALTypeCode(image_data.dtype))

    if output_dataset is None:
        print("Failed to create the output TIFF file.")
        return

    # Write the image data to the raster bands
    for i in range(3):
        output_band = output_dataset.GetRasterBand(i + 1)
        output_band.WriteArray(image_data[i])

    # Set the metadata from the original GeoTIFF file
    output_dataset.SetProjection(original_dataset.GetProjection())
    output_dataset.SetGeoTransform(original_dataset.GetGeoTransform())

    # Close the datasets
    original_dataset = None
    output_dataset = None


def convert_to_tiff(original_tiff_path, output_tiff_path, image_convert):

    # Read the GeoTIFF image
    image_data, num_bands, width, height = read_tiff_image(original_tiff_path)

    if image_data is not None:
        # Resize the JPEG image to match the size of the GeoTIFF image
        resized_image = cv2.resize(image_convert, (width, height))

        # Convert the BGR image to RGB
        resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        # Convert the image to a NumPy array
        modified_image_data = resized_image_rgb.transpose(2, 0, 1)

        # Save the modified image data as a new GeoTIFF file with the original metadata
        save_tiff_image(modified_image_data, output_tiff_path, original_tiff_path)



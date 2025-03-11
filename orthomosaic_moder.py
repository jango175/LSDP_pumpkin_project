import rasterio
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rasterio.plot import show


def show_image(window_name: str, img: cv2.typing.MatLike) -> None:
    """
    Show the image in the full screen window.

    :param window_name: The name of the window.
    :param img: The image to show.
    """

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(window_name, img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def read_orthomosaic(file_path: str) -> tuple[np.ndarray, dict, rasterio.profiles.Profile]:
    """
    Read an orthomosaic image using Rasterio.

    :param file_path: Path to the orthomosaic image file.

    :return: Tuple containing the image data, metadata and profile.
    """

    with rasterio.open(file_path) as src:
        # read the image data
        image = src.read()
        # get metadata
        meta = src.meta

        # print some basic info
        print(f"Image shape: {image.shape}")
        print(f"Coordinate system: {src.crs}")

        return image, meta, src.profile


def convert_to_opencv(image: np.ndarray) -> np.ndarray:
    """
    Convert to format suitable for Opencv2.

    :param image: Image data in rasterio format.

    :return: Image data in OpenCV format.
    """

    # rasterio reads as (bands, height, width) but OpenCV expects (height, width, bands)
    if image.shape[0] in [1, 3, 4]: # if band dimension is first
        image = np.transpose(image, (1, 2, 0))

    # if single band, expand to 3 channels for easier visualization
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)

    # handle 4-channel images (RGBA)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    else:
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)

    # normalize to 0-255 if image is float
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        image = image.astype(np.uint8)

    return image


def count_pumpkins(img: cv2.typing.MatLike) -> int:
    """
    Count pumpkins in the image.

    :param img: The image with pumpkins.

    :return: The number of pumpkins.
    """

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for the color orange in HSV
    # lower_orange = (0, 85, 234)
    # upper_orange = (55, 228, 255)
    lower_orange = (12, 100, 180)
    upper_orange = (27, 230, 255)

    # Create a mask for the orange color
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Apply the mask to the image
    masked = cv2.bitwise_and(img, img, mask=mask)

    # Show the masked image
    # show_image('Masked', masked)

    # Convert the masked image to grayscale
    gray_masked = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Apply the Gaussian blur
    blurred = cv2.GaussianBlur(gray_masked, (11, 11), 0)

    # Apply the threshold
    _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)

    # # # Apply the morphological operations
    # thresh = cv2.dilate(thresh, None, iterations=6)
    # thresh = cv2.erode(thresh, None, iterations=6)

    # Show the masked image
    # show_image('Thresh', thresh)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    # Show the image with contours
    show_image('Contours', img)

    # Correct the number of the pumpkins which are close to each other
    area = []
    for contour in contours:
        # check the contour dimension
        area.append(cv2.contourArea(contour))

    single_pumpkin_area = np.median(area) # 150

    # print(f'areas: {area}\n')
    print(f'single pumpkin area: {single_pumpkin_area}\n')

    # Count pumpkins
    number_of_pumpkins = len(contours)
    print(f'Number of pumpkins before correction: {number_of_pumpkins}\n')

    prev_number_of_pumpkins = number_of_pumpkins

    for a in area:
        divider = a // single_pumpkin_area
        i = 1

        while divider >= 1:
            divider = a // (i*single_pumpkin_area)
            i += 1

        i -= 1

        for j in range(i, 0, -1):
            if a > 1.75*j*single_pumpkin_area:
                number_of_pumpkins += j - 1
                break

    print(f'Corrected pumpkins: {number_of_pumpkins - prev_number_of_pumpkins}\n')

    return number_of_pumpkins


def main() -> None:
    """
    Main function.
    """

    # path to orthomosaic file
    file_path = "orthomosaic.tif"

    # read the file
    image, meta, profile = read_orthomosaic(file_path)

    # process with OpenCV
    img = convert_to_opencv(image)

    # count pumpkins
    count = count_pumpkins(img)
    print(f"Number of pumpkins: {count}")

    # visualize results
    show_image("Orthomosaic", img)


if __name__ == "__main__":
    main()

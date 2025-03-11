import cv2 as cv
import os
import matplotlib.pyplot as plt
import numpy as np


def show_image(window_name: str, img: cv.typing.MatLike) -> None:
    """
    Show the image in the full screen window.

    :param window_name: The name of the window.
    :param img: The image to show.
    """

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    cv.imshow(window_name, img)
    cv.waitKey(0)

    cv.destroyAllWindows()


def image_histogram(img: cv.typing.MatLike) -> None:
    """
    Draw the RGB histogram of the image.

    :param img: The image.
    """

    # Split the image into its respective channels
    channels = cv.split(img)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title('RGB Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')

    # Calculate and plot the histogram for each channel
    for (channel, color) in zip(channels, colors):
        hist = cv.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])

    plt.grid()
    plt.show()


def count_pumpkins(img: cv.typing.MatLike) -> int:
    """
    Count pumpkins in the image.

    :param img: The image with pumpkins.

    :return: The number of pumpkins.
    """

    # Convert the image to HSV color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Define the range for the color orange in HSV
    # lower_orange = (0, 85, 234)
    # upper_orange = (55, 228, 255)
    lower_orange = (12, 100, 180)
    upper_orange = (27, 230, 255)

    # Create a mask for the orange color
    mask = cv.inRange(hsv, lower_orange, upper_orange)

    # Apply the mask to the image
    masked = cv.bitwise_and(img, img, mask=mask)

    # Show the masked image
    # show_image('Masked', masked)

    # Convert the masked image to grayscale
    gray_masked = cv.cvtColor(masked, cv.COLOR_BGR2GRAY)

    # Apply the Gaussian blur
    blurred = cv.GaussianBlur(gray_masked, (9, 9), 0)

    # Apply the threshold
    _, thresh = cv.threshold(blurred, 10, 255, cv.THRESH_BINARY)

    # # # Apply the morphological operations
    # thresh = cv.dilate(thresh, None, iterations=6)
    # thresh = cv.erode(thresh, None, iterations=6)

    # Show the masked image
    # show_image('Thresh', thresh)

    # Find contours
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Draw contours
    cv.drawContours(img, contours, -1, (255, 0, 0), 1)

    # Show the image with contours
    show_image('Contours', img)

    # Correct the number of the pumpkins which are close to each other
    area = []
    for contour in contours:
        # check the contour dimension
        area.append(cv.contourArea(contour))

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
            if a > 1.5*j*single_pumpkin_area:
                number_of_pumpkins += j - 1
                break

    print(f'Corrected pumpkins: {number_of_pumpkins - prev_number_of_pumpkins}\n')

    return number_of_pumpkins


def main() -> None:
    """
    The main function.
    """

    # # List images
    # img_dir = 'pumpkin_images'
    # images_list = os.listdir(img_dir)
    # print(f'Number of images in the {img_dir}: {len(images_list)}')
    # img = cv.imread(os.path.join(img_dir, images_list[180]))

    # Load the image
    # img = cv.imread('orthomosaic.png')
    img = cv.imread('orthomosaic_cropped.png')

    # Show the image
    # show_image('Pumpkin', img)

    # Show the histogram
    # image_histogram(img)

    pumpkin_cnt = count_pumpkins(img)
    print(f'Number of pumpkins: {pumpkin_cnt}')


if __name__ == '__main__':
    main()
else:
    print('Imported pumpkin_counter.py')

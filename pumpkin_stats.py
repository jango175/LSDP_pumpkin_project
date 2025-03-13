import cv2
from matplotlib import pyplot as plt


def image_cielab_histogram(img: cv2.typing.MatLike) -> None:
    """
    Draw the CIELAB histogram of the image.

    :param img: The image.
    """

    # Convert the image to the CIELAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the image into its respective channels
    channels = cv2.split(lab)
    colors = ('black', 'green', 'blue')
    plt.figure()
    plt.title('CIELAB Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')

    # Calculate and plot the histogram for each channel
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.legend(['L*', 'a*', 'b*'])
    plt.grid()
    plt.show()



def image_hsv_histogram(img: cv2.typing.MatLike) -> None:
    """
    Draw the HSV histogram of the image.

    :param img: The image.
    """

    # Convert the image to the HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split the image into its respective channels
    channels = cv2.split(hsv)
    colors = ('black', 'orange', 'yellow')
    plt.figure()
    plt.title('HSV Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')

    # Calculate and plot the histogram for each channel
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.legend(['Hue', 'Saturation', 'Value'])
    plt.grid()
    plt.show()


def image_rgb_histogram(img: cv2.typing.MatLike) -> None:
    """
    Draw the RGB histogram of the image.

    :param img: The image.
    """

    # Split the image into its respective channels
    channels = cv2.split(img)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title('RGB Histogram')
    plt.xlabel('Bins')
    plt.ylabel('# of Pixels')

    # Calculate and plot the histogram for each channel
    for (channel, color) in zip(channels, colors):
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.legend(['Blue', 'Green', 'Red'])
    plt.grid()
    plt.show()


def image_stats(img: cv2.typing.MatLike) -> None:
    """
    Calculate the mean and standard deviation of the image.

    :param img: The image.
    """

    # Calculate the mean and standard deviation of the image in the CIELAB color space
    (l, a, b) = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
    (l_mean, l_std) = (l.mean(), l.std())
    (a_mean, a_std) = (a.mean(), a.std())
    (b_mean, b_std) = (b.mean(), b.std())

    # Print the mean and standard deviation of the image
    print('\nCIELAB Colour Space:')
    print(f'L*: mean={l_mean}, std={l_std}')
    print(f'a*: mean={a_mean}, std={a_std}')
    print(f'b*: mean={b_mean}, std={b_std}\n')

    # Do the same in HSV color space
    (h, s, v) = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    (h_mean, h_std) = (h.mean(), h.std())
    (s_mean, s_std) = (s.mean(), s.std())
    (v_mean, v_std) = (v.mean(), v.std())

    # Print the mean and standard deviation of the image
    print('HSV Colour Space:')
    print(f'H: mean={h_mean}, std={h_std}')
    print(f'S: mean={s_mean}, std={s_std}')
    print(f'V: mean={v_mean}, std={v_std}\n')

    # And in the RGB color space
    (b, g, r) = cv2.split(img)
    (r_mean, r_std) = (r.mean(), r.std())
    (g_mean, g_std) = (g.mean(), g.std())
    (b_mean, b_std) = (b.mean(), b.std())

    # Print the mean and standard deviation of the image
    print('RGB Colour Space:')
    print(f'R: mean={r_mean}, std={r_std}')
    print(f'G: mean={g_mean}, std={g_std}')
    print(f'B: mean={b_mean}, std={b_std}\n')


def main(file_path) -> None:
    """
    Main function to demonstrate the image histogram and statistics functions.

    :param file_path: The path to the image.
    """

    img = cv2.imread(file_path)

    image_rgb_histogram(img)
    image_hsv_histogram(img)
    image_cielab_histogram(img)

    image_stats(img)


# Entry point
if __name__ == "__main__":
    file_path = 'pumpkin_example.JPG'

    main(file_path)
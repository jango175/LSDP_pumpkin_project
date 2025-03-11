import cv2
from matplotlib import pyplot as plt


def image_histogram(img: cv2.typing.MatLike) -> None:
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

    plt.grid()
    plt.show()


def main()-> None:
    img = cv2.imread('pumpkin_example.JPG')
    image_histogram(img)


if __name__ == "__main__":
    main()
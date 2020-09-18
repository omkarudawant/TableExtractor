import cv2
import numpy as np


class PreprocessImage:
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def grayscale(self, image: np.array) -> np.array:
        if type(image) == str:
            image = cv2.imread(image, 0)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    # noinspection PyMethodMayBeStatic
    def remove_noise(self, image: np.array) -> np.array:
        if type(image) == str:
            image = cv2.imread(image, 0)

        image = cv2.medianBlur(image, 2)
        return image

    # noinspection PyMethodMayBeStatic
    def thresholding(self, image: np.array) -> np.array:
        if type(image) == str:
            image = cv2.imread(image, 0)

        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return image

    # noinspection PyMethodMayBeStatic
    def dilate(self, image: np.array) -> np.array:
        if type(image) == str:
            image = cv2.imread(image, 0)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.dilate(image,
                          kernel,
                          iterations=1)

    # noinspection PyMethodMayBeStatic
    def erode(self, image: np.array) -> np.array:
        if type(image) == str:
            image = cv2.imread(image, 0)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.erode(image, kernel, iterations=1)

    # noinspection PyMethodMayBeStatic
    # opening - erosion followed by dilation
    def opening(self, image):
        if type(image) == str:
            image = cv2.imread(image, 0)
        kernel = np.ones((5, 5), np.uint8)
        return cv2.morphologyEx(image,
                                cv2.MORPH_OPEN,
                                kernel)

    # noinspection PyMethodMayBeStatic
    def canny(self, image):
        if type(image) == str:
            image = cv2.imread(image, 0)
        return cv2.Canny(image, 100, 200)

    # noinspection PyMethodMayBeStatic
    # skew correction
    def deskew(self, image):
        if type(image) == str:
            image = cv2.imread(image, 0)

        coords = np.column_stack(
            np.where(image > 0)
        )
        angle = cv2.minAreaRect(coords)[-1]

        angle = -(90 + angle) if angle < -45 else -angle

        (h, w) = image.shape  # Assuming grayscale image
        center = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D(
            center, angle, 1.0
        )
        rotated = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return rotated

import cv2
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI, RIL


def run_ocr(ocr_img: np.array):
    ocr_img = Image.fromarray(ocr_img)
    with PyTessBaseAPI() as api:
        api.SetImage(ocr_img)
        boxes = api.GetComponentImages(RIL.WORD, True)
        print(f'Found {len(boxes)} components')
        for i, (im, box, _, _) in enumerate(boxes):
            (x, y, w, h) = box['x'], box['y'], box['w'], box['h']

            if (w * h) > 50:
                ocr_img = cv2.rectangle(ocr_img, (x, y), (x + w, y + h),
                                        (0, 255, 0), 2)
    return np.array(ocr_img)


def find_contours(cnt_img: np.array) -> np.array:
    thresholded = cv2.threshold(cnt_img, 0, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((3, 3), np.int8)
    dilated = cv2.dilate(thresholded, kernel, iterations=1)
    contours = cv2.findContours(
        dilated,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_TC89_KCOS
    )[0]
    print(f'Found {len(contours)} components')
    components = dict()
    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)
        if 400 > (w * h) > 150:
            start_x, start_y = x - 3, y - 3
            width, height = x + w + 3, y + h + 3
            a = np.array((25, 10))
            b = np.array((start_x, start_y))
            distance = np.sqrt(np.dot(a - b, a - b))

            components[distance] = (start_x, start_y, width, height)

    comp = {
        k: v for (k, v) in
        sorted(
            components.items(),
            key=lambda l: [l[0], l[1][1]],
            reverse=False
        )
    }

    for i, v in comp.items():
        print(i, v)

        table_image = cv2.rectangle(
            cnt_img, (v[0], v[1]), (v[2], v[3]),
            (0, 255, 0), 2
        )
        cv2.imshow('sample', cnt_img[v[1]:v[3], v[0]:v[2]])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return cnt_img

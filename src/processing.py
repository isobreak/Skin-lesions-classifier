import cv2
import numpy
import numpy as np


def get_tumor_contour(img: numpy.ndarray,  solve_hair: bool = True,
                      area_ratio_thresh: float = 0.65,
                      indent_ratio_thresh: float = 0.18) -> numpy.ndarray:
    """
    :param img: the source image\n
    :param solve_hair: if hair-problem should be solved (decreases quality of segmentation, but ignores hair)
    :param area_ratio_thresh: max suspicious_area/total_area ratio of object to be considered as tumor\n
    :param indent_ratio_thresh: max x_indent/width (or y_indent/height) ratio of object  to be considered as tumor\n
    :return: tumor contour"""

    def get_k(img: np.ndarray, intensity_thresh: int = 145) -> float:
        """Returns coefficient for exposition filter matrix corresponding given target intensity\n
        - img - source image
        - target_intensity - average intensity of resulting image"""
        def calc_intensity(img):
            """Returns average intensity of image"""
            flatten = img.flatten()
            sum = np.sum(flatten)
            return sum / len(flatten)

        k = 0.3
        while k < 2.2:
            k += 0.05
            kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]]) * k
            intensity = calc_intensity(cv2.filter2D(img, -1, kernel))
            if intensity > intensity_thresh:
                return k

        return k

    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]
    AREA_THRESH = area_ratio_thresh * HEIGHT * WIDTH
    X_INDENT_LEFT = indent_ratio_thresh * WIDTH
    X_INDENT_RIGHT = WIDTH - X_INDENT_LEFT
    Y_INDENT_UPPER = indent_ratio_thresh * HEIGHT
    Y_INDENT_LOWER = HEIGHT - Y_INDENT_UPPER

    best_contours = []
    orig = img

    for mode in range(3):
        if mode == 0:
            img = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        elif mode == 1:
            img, _, _ = cv2.split(orig)
        elif mode == 2:
            img_b, img_g, _ = cv2.split(orig)
            img = cv2.merge([img_b, img_g, img_b])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        k = get_k(img, intensity_thresh=140)
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]]) * k
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.medianBlur(img, 15)

        _, mask = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY_INV)
        if solve_hair is True:
            open_kernel = np.ones((9, 9))
            mask = cv2.erode(mask, open_kernel, iterations=5)
            mask = cv2.dilate(mask, open_kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        max_area = 0
        i_max = -1
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > max_area and area <= AREA_THRESH:
                M = cv2.moments(contour)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if X_INDENT_LEFT < cx < X_INDENT_RIGHT and Y_INDENT_UPPER < cy < Y_INDENT_LOWER:
                    max_area = area
                    i_max = i

        if i_max != -1:
            best_contours.append(contours[i_max])

    if len(best_contours) == 0:
        return None
    else:
        best_ratio = cv2.arcLength(best_contours[0], True) / np.sqrt(cv2.contourArea(best_contours[0]))
        best_mode = 0
        for mode, contour in enumerate(best_contours):
            ratio = cv2.arcLength(contour, True) / np.sqrt(cv2.contourArea(contour))
            if ratio <= best_ratio:
                best_ratio = ratio
                best_mode = mode
    res_contour = best_contours[best_mode]
    res_contour = cv2.convexHull(res_contour)

    return res_contour


def get_cropped_image(img: numpy.ndarray, size_step: int = 224, const_res: bool = True, solve_hair: bool = True,
                      apply_mask: bool = False, draw_contour: bool = False,
                      contour_color: tuple[int, int, int] = (255, 0, 0),
                      area_ratio_thresh: float = 0.65, indent_ratio_thresh: float = 0.18, contour = None) -> numpy.ndarray:
    """
    Returns cropped image of (N*size_step, N*size_step) resolution, where N is a min appropriate number.
    Optionally applies a mask to the image and/or draws a contour on it
    :param img: source image
    :param size_step: size increase step (equals to min of HEIGHT and WIDTH if is set up to 0 or less)
    :param const_res: resize all images to (size_step, size_step) regardless their resolution after processing
    :param solve_hair: whether hair-problem should be solved (decreases quality of segmentation,
    but more likely ignores hair)
    :param apply_mask: apply mask to the image
    :param draw_contour: draw selected contour
    :param contour_color: color of drawn contour
    :param area_ratio_thresh: max suspicious_area/total_area ratio of object to be considered as tumor
    :param indent_ratio_thresh: max x_indent/width (or y_indent/height) ratio of object to be considered as tumor
    :param contour: existing contour to use except calculating a new one
    :return cropped image of the minimal of the available discrete sizes:
    """
    HEIGHT = img.shape[0]
    WIDTH = img.shape[1]

    if contour is None:
        contour = get_tumor_contour(img, solve_hair=solve_hair, area_ratio_thresh=area_ratio_thresh,
                                    indent_ratio_thresh=indent_ratio_thresh)

    if contour is not None:
        if apply_mask and contour is not None:
            mask = np.zeros([img.shape[0], img.shape[1]], dtype='uint8')
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
            img = cv2.bitwise_and(img, img, mask=mask)
        if draw_contour and contour is not None:
            cv2.drawContours(img, [contour], -1, color=contour_color, thickness=2)

        x, y, w, h = cv2.boundingRect(contour)
    else:
        cont_size = min(HEIGHT, WIDTH)
        x, y, w, h = (WIDTH - cont_size) // 2, (HEIGHT - cont_size) // 2, cont_size, cont_size

    if size_step <= 0:
        size_step = min(WIDTH, HEIGHT)
    size = size_step
    while w > size or h > size:
        size += size_step
        if size > HEIGHT or size > WIDTH:
            size -= size_step
            w -= size_step
            break

    x_left = (size - w) // 2
    x_right = size - w - x_left
    y_upper = (size - h) // 2
    y_lower = size - h - y_upper
    y1 = y - y_upper
    y2 = y + h + y_lower
    x1 = x - x_left
    x2 = x + w + x_right

    if x1 < 0:
        shift = 0 - x1
        x1 += shift
        x2 += shift
    elif x2 > WIDTH:
        shift = x2 - WIDTH
        x1 -= shift
        x2 -= shift
    if y1 < 0:
        shift = 0 - y1
        y1 += shift
        y2 += shift
    elif y2 > HEIGHT:
        shift = y2 - HEIGHT
        y1 -= shift
        y2 -= shift

    img = img[y1:y2, x1:x2]
    if const_res:
        img = cv2.resize(img, (size_step, size_step))

    return img


if __name__ == "__main__":
    pass

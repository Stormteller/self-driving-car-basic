import cv2
import os
import matplotlib.image as mpimg
import numpy as np
from uuid import uuid4
from datetime import datetime


def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def gray_scale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def canny_edge_detection(img, low_threshold=50, high_threshold=150):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    return cv2.bitwise_and(img, mask)


def line_angle(line):
    x1, y1, x2, y2 = line
    return np.rad2deg(np.arctan2((y2 - y1), (x2 - x1)))


def hough_lines(img, rho=1, theta=np.pi/180, threshold=20, min_line_len=20, max_line_gap=30):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)


def averaging_lines(lines, horizontal_threshold=1):
    if not isinstance(lines, np.ndarray):
        return [[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]

    left_lines = [line for line in lines if line_angle(line[0]) < -horizontal_threshold and (line[0][0] < 160 or line[0][2] < 160)]
    right_lines = [line for line in lines if line_angle(line[0]) > horizontal_threshold and (line[0][0] > 160 or line[0][2] > 160)]
    horizontal_lines = [line for line in lines if -horizontal_threshold < line_angle(line[0]) < horizontal_threshold]

    return [[averaging_line(left_lines),
             averaging_line(horizontal_lines),
             averaging_line(right_lines)]]


def averaging_line(lines):
    if not lines:
        return np.array([0, 0, 0, 0], dtype=np.int32)

    coords_number = 4

    ave_coords = [np.average([line[0][i] for line in lines]) for i in range(coords_number)]
    return np.array(ave_coords, dtype=np.int32)


def draw_lines(lines, img_shape, color=(255, 0, 0), thickness=1):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    line_img = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for all_lines in lines:
        for x1, y1, x2, y2 in all_lines:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return line_img


def line_length(line):
    x1, y1, x2, y2 = line
    return np.sqrt(np.power(x2 - x1, 2) + np.power(y2 - y1, 2))


def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def get_road_lines_from_image(input_image):
    image = gray_scale(input_image)
    image = gaussian_blur(image, 11)
    low_threshold = 49
    image = canny_edge_detection(image, low_threshold=low_threshold, high_threshold=low_threshold*3)
    vertices = np.array([[[0, 120], [60, 70], [240, 70], [320, 120]]], dtype=np.int32)
    image = region_of_interest(image, vertices)
    lines = hough_lines(image, min_line_len=20)
    averaged_lines = averaging_lines(lines, horizontal_threshold=3)

    img_to_save = weighted_img(draw_lines(averaged_lines, image.shape, thickness=3), input_image)
    cv2.imwrite(r'D:\Google Drive\Diploma\project\imgs\img_{}.png'.format(uuid4()), img_to_save)
    cv2.imwrite(r'D:\Google Drive\Diploma\project\imgs\img_{}.png'.format(uuid4()), image)

    cleaned_lines = list(map(lambda x: x if line_length(x) > 0 else None, averaged_lines[0]))
    return {
        'lines': {
            'left': cleaned_lines[0],
            'horizontal': cleaned_lines[1],
            'right': cleaned_lines[2]
        },
        'angels': {
            'left': line_angle(averaged_lines[0][0]),
            'horizontal': line_angle(averaged_lines[0][1]),
            'right': line_angle(averaged_lines[0][2])
        }
    }

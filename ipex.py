import os
import sys
import math
import shutil
import logging
import numpy
from numpy.typing import NDArray
import cv2 as cv
from pathlib import Path
from datetime import datetime

SCRIPT_NAME = 'IPEX : Image Paper EXtractor'
VERSION = '1.2.0'

RESULT_IMAGE_EXT = '.png' # Can be any type handled by OpenCV, see documentation for valid values.
DETECTION_IMAGE_MAX_DIM = 1024 # In pixels, if the lagest dimension (width or height) of the input image is
#                                bigger than this value the image will be downscale ONLY for paper detect calculations.
#                                Smaller value mean faster computation but less accuracy.
KERNEL_ERODE_SIZE = 3 # Size in pixels of the "brush" for the erode operation, value need to be odd (3, 5, 7, ...)
SIMPLIFIED_CONTOUR_MAX_COEF = 0.15 # Maximum ratio of simplification allowed for the contour point reduction (e.g. simplify_contour function)
PAPER_DEFORMATION_TOLERANCE = 0.01 # Above this value a complexe method will be used to compute paper aspect ratio
WHITE_CORRECTION_FACTOR = 0.2 # Interval [0.0, 1.0], a bigger value remove small details but improve white background for documents

DEBUG = False

class ColoredFormatter(logging.Formatter):
    """Custom formatter handling color"""
    cyan = '\x1b[36;20m'
    green = '\x1b[32;20m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'
    message_format = '%(levelname)-8s - %(message)s'

    FORMATS = {
        logging.DEBUG: cyan + message_format + reset,
        logging.INFO: green + message_format + reset,
        logging.WARNING: yellow + message_format + reset,
        logging.ERROR: red + message_format + reset,
        logging.CRITICAL: bold_red + message_format + reset
    }

    def format(self, record):
        log_format = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)

def init_logger()-> logging.Logger:
    """Initialize script logger"""

    logger_name = Path(__file__).stem
    logger = logging.getLogger(logger_name)
    logger.setLevel(level = logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level=logging.DEBUG)
    console_handler.setFormatter(ColoredFormatter())

    logger.addHandler(console_handler)

    return logger

LOG = init_logger()

def delete_results_folder()-> bool:
    if PATH_DIR_RESULTS and PATH_DIR_RESULTS.exists():
        try:
            shutil.rmtree(PATH_DIR_RESULTS)
        except Exception as e:
            LOG.error('Failed to delete the empty results folder. Reason: {}'.format(e))
            return False
    return True

def create_results_folder()-> bool:
    working_directory = Path.cwd()

    # Step 1: Create the directory path for the results and set it as a global variable
    global PATH_DIR_RESULTS
    PATH_DIR_RESULTS = working_directory.joinpath('results', datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # Step 2: Creating the directories
    try:
        PATH_DIR_RESULTS.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        LOG.error('Failed to create results folder. Reason: {}'.format(e))
        return False

    return True

def save_to_results_folder(paper: NDArray[numpy.uint8], filename: str):
    cv.imwrite(str(PATH_DIR_RESULTS.joinpath(filename + RESULT_IMAGE_EXT)), paper)

def downscale_image(image: NDArray[numpy.uint8])-> tuple[float, NDArray[numpy.uint8]]:
    factor = 1.0
    height, width = image.shape[:2]

    # Step 1: If image doesn't need resize do nothing
    if height <= DETECTION_IMAGE_MAX_DIM and width <= DETECTION_IMAGE_MAX_DIM:
        return (1.0, image)

    # Step 2: Determine the biggest dimension between height and width
    if height > width:
        # Step 3: Compute the new dimension, scaling by reduction factor
        factor = (float(DETECTION_IMAGE_MAX_DIM) / float(height))
        width = int(float(width) * factor)
        height = DETECTION_IMAGE_MAX_DIM
    else:
        # Step 3: Compute the new dimension, scaling by reduction factor
        factor = (float(DETECTION_IMAGE_MAX_DIM) / float(width))
        height = int(float(height) * factor)
        width = DETECTION_IMAGE_MAX_DIM

    # Step 4: Resize and return the new image
    return (factor, cv.resize(image, (width, height), interpolation=cv.INTER_AREA))

def split_grayscale(image: NDArray[numpy.uint8])-> tuple[float, NDArray[numpy.uint8]]:
    if len(image.shape) >= 3:
        YUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)
        return (YUV[:, :, 0], YUV)
    return (image, image)

def auto_brightness_and_contrast(image: NDArray[numpy.uint8], clip_hist_percent: int = 1)-> NDArray[numpy.uint8]:
    gray, _ = split_grayscale(image)

    # Step 2: Calculate grayscale histogram
    hist = cv.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)

    # Step 3: Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))

    # Step 4: Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0

    # Step 5: Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Step 6: Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Step 7: Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    # Step 8: Appling and returning the corrected image
    return cv.convertScaleAbs(image, alpha=alpha, beta=beta)

def convert_to_binary_image(image: NDArray[numpy.uint8])-> NDArray[numpy.uint8]:
    # Step 1: Create the kernels, this is like brushes in painting
    kernel_type = cv.MORPH_CROSS# Seems to give the best results
    kernel_morph_size = 5# Can be adjusted, or maybe computed
    kernel_erode_size = KERNEL_ERODE_SIZE# Need to be small to avoid create issues
    kernel_morph = cv.getStructuringElement(kernel_type, (kernel_morph_size, kernel_morph_size))
    kernel_erode = cv.getStructuringElement(kernel_type, (kernel_erode_size, kernel_erode_size))

    # Step 2: Create the first layer of the binary image using thresholding method
    layer_1 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    # Step 2.B: Applying a morphological opening & closing to remove useless blobs (optional)
    layer_1 = cv.morphologyEx(layer_1, cv.MORPH_OPEN, kernel_morph)
    layer_1 = cv.morphologyEx(layer_1, cv.MORPH_CLOSE, kernel_morph)

    # Step 3: Create the second layer using adaptive thresholding method
    layer_2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 3)
    # Step 3.B: Applying a morphological closing to avoid discontinuated shapes
    layer_2 = cv.morphologyEx(layer_2, cv.MORPH_CLOSE, kernel_morph)

    # Step 4: Merging the two layer by doing a substraction
    image = layer_1 - layer_2
    # Step 4.B: Clamping values with a threshold operation, required to ensure it's a valid binary image
    image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]

    # Step 5: Erode to ensure shapes are properly separated
    image = cv.erode(image, kernel_erode)

    # Step 6: Last pass to remove small blobs (optional)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel_morph)

    return image

def scale_contour_from_centroid(contour: NDArray[numpy.float32], scale: float)-> NDArray[numpy.float32]:
    # Step 1: Determine the centroid of the contour
    moment = cv.moments(contour)
    center_x = int(moment['m10'] / moment['m00'])
    center_y = int(moment['m01'] / moment['m00'])

    # Step 2: move the contour center at 0,0
    contour_normalized = contour - [center_x, center_y]

    # Step 3: Scale
    contour = contour_normalized * scale

    # Step 4: Move back the contour to it position
    contour = contour + [center_x, center_y]

    return contour

def simplify_contour_compute_weight(contour: NDArray[numpy.float32], index: int)-> float:
    p1 = contour[(index-1)%contour.shape[0]][0]
    p2 = contour[index][0]
    p3 = contour[(index+1)%contour.shape[0]][0]
    return (0.5 * abs((p1[0] * (p2[1] - p3[1])) + (p2[0] * (p3[1] - p1[1])) + (p3[0] * (p1[1] - p2[1]))))

def simplify_contour(contour: NDArray[numpy.float32], nbr_ptr_limit: int = 4)-> NDArray[numpy.float32]:
    # Using a naive version of Visvalingam-Whyatt simplification algorithm

    # points_weights will be used to determine the importance of points,
    # in the Visvalingam-Whyatt algorithm it's the area of the triangle created by a point and his direct neighbours
    points_weights = numpy.zeros(contour.shape[0])

    # Step 1: First pass, computing all points weight
    for index in range(contour.shape[0]):
        points_weights[index] = simplify_contour_compute_weight(contour, index)

    # Step 2: Until we have 4 points we delete the less significant point and iterate
    while contour.shape[0] > nbr_ptr_limit:
        # Step 2.A: Get point index with minimum weight
        index_pnt = numpy.argmin(points_weights)

        # Step 2.B: Remove it
        contour = numpy.delete(contour, index_pnt, axis=0)
        points_weights = numpy.delete(points_weights, index_pnt)
        if contour.shape[0] == nbr_ptr_limit:
            break

        # Step 2.C: Re-compute neighbours points weight
        index_pnt_prev = (index_pnt-1)%contour.shape[0]
        index_pnt_next = (index_pnt)%contour.shape[0]
        points_weights[index_pnt_prev] = simplify_contour_compute_weight(contour, index_pnt_prev)
        points_weights[index_pnt_next] = simplify_contour_compute_weight(contour, index_pnt_next)

    return contour

def find_paper_contour_from_binary_image(image: NDArray[numpy.uint8])-> NDArray[numpy.float32]:
    # Step 1: Find contours of shapes in the binary image
    contours = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

    # Step 2: Sort and choose the largest contour
    contours = sorted(contours, key = cv.contourArea)
    contour = contours[-1]

    # Step 3: Getting the shape from the contour
    contour = cv.convexHull(contour)
    cnt_points = len(contour)
    arclen = cv.arcLength(contour, True)
    if cnt_points > 4:
        contour = cv.approxPolyDP(contour, 0.02 * arclen, True)
        cnt_points = len(contour)

    if cnt_points < 4 or not cv.isContourConvex(contour):
        # The best shape candidate seems to not be a rectangle
        return None

    # Step 3.B: Try to simplify the contour to 4 points
    if cnt_points > 4:
        area_previous = cv.contourArea(contour)
        simplified_contour = simplify_contour(contour)
        area_simplified = cv.contourArea(simplified_contour)

        # Check if simplified_contour is convex and the diffference in area is acceptable
        if cv.isContourConvex(simplified_contour) and (1.0 - (area_previous / area_simplified)) <= SIMPLIFIED_CONTOUR_MAX_COEF:
            contour = simplified_contour
        else:
            return None

    # Step 4: Scale the shape since when generating the binary map the shape has been eroded
    pixels_to_add = int(KERNEL_ERODE_SIZE / 2) * 2
    contour_side_length = cv.norm(contour[0][0], contour[1][0])
    scale = (contour_side_length + float(pixels_to_add)) / contour_side_length
    contour = scale_contour_from_centroid(contour, scale)

    return contour

def retrieve_contour(image: NDArray[numpy.uint8], paper_index: int)-> NDArray[numpy.float32]:
    if DEBUG:
        original = image.copy()
        _, original = downscale_image(original)

    # Step 1: Convert image to grayscale (Using Y of YUV is better than GRAY)
    image, _ = split_grayscale(image)
    if DEBUG:
        save_to_results_folder(image, 'Paper-{:03d}_DEBUG_01_Grayscale'.format(paper_index))

    # Step 2: Downscale the image if necessary, save the factor
    downscale_factor, image = downscale_image(image)
    if DEBUG:
        save_to_results_folder(image, 'Paper-{:03d}_DEBUG_02_Downscale'.format(paper_index))

    # Step 3: Median blur the image (a.k.a Noise Median Reduction),
    # this help removing the unecessary details while conserving the edges
    aperture_linear_size = 9
    image = cv.medianBlur(image, aperture_linear_size)
    if DEBUG:
        save_to_results_folder(image, 'Paper-{:03d}_DEBUG_03_Median-Blur'.format(paper_index))

    # Step 4: Correct the brightness and contrast (optional)
    image = auto_brightness_and_contrast(image, 5)
    if DEBUG:
        save_to_results_folder(image, 'Paper-{:03d}_DEBUG_04_Contrast'.format(paper_index))

    # Step 5: Convert to binary map (threshold & morphological transformations)
    image = convert_to_binary_image(image)
    if DEBUG:
        save_to_results_folder(image, 'Paper-{:03d}_DEBUG_05_Treshold'.format(paper_index))

    # From this point in the process there is two very different ways:
    # 1) Using cv.findContours, extracting the biggest contour, check the shape to see it's a rectangle
    # 2) Using cv.Canny then cv.HoughLines, to find lines, check intersections and try to find shapes
    #
    # The first way is easier and work well most of the time, the second one is harder, more codes but can
    # work better in some complexe situations than the first one.
    # Here I will use cv.findContours

    # Step 6: Find the paper contour
    contour = find_paper_contour_from_binary_image(image)
    if contour is None:
        return None

    if DEBUG:
        cv.drawContours(original, [contour.astype(numpy.int32)], -1, (0, 0, 255), 1, cv.LINE_AA)
        save_to_results_folder(original, 'Paper-{:03d}_DEBUG_06_Contour'.format(paper_index))

    # Step 7: Very important! Apply the downscale factor to scale up the contour to the correct size
    contour = (contour * (1.0 / downscale_factor))

    return contour

def get_corners_from_coutour(contour: NDArray[numpy.float32])-> NDArray[numpy.float32]:
    # We need first to ensure a clockwise orientation for the contour
    corners = None

    # Step 1: Find top left point, using distance to top left of the picture
    dist_list = [[numpy.linalg.norm(point[0]), index] for index, point in enumerate(contour)]
    dist_list = sorted(dist_list, key = lambda x: x[0])

    index_pnt_tl = dist_list[0][1]

    # Step 2: Find the others points order. Since the contour has been retrieved via 
    #         cv.findContours it's either sorted in clockwise or counter clockwise,
    count_points = 4# We know at this point that the contour as only 4 points, no more, no less
    index_pnt_prev = (index_pnt_tl-1)%count_points
    index_pnt_next = (index_pnt_tl+1)%count_points
    index_pnt_last = (index_pnt_tl+2)%count_points
    # Step 2.B: Comparing x axis values of the neighbours of the top left point find out if the
    #           contour has been sorted in clockwise or counter clockwise
    if contour[index_pnt_prev][0][0] > contour[index_pnt_next][0][0]:
        # Counter clockwise
        corners = numpy.array([contour[index_pnt_tl][0],
                                contour[index_pnt_prev][0],
                                contour[index_pnt_last][0],
                                contour[index_pnt_next][0]])
    else:
        # Clockwise
        corners = numpy.array([contour[index_pnt_tl][0],
                                contour[index_pnt_next][0],
                                contour[index_pnt_last][0],
                                contour[index_pnt_prev][0]])

    # Step 3: Convert array to int
    #corners = numpy.rint(corners).astype(int)

    return corners

def compute_aspect_ratio(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32])-> float:
    # Based on :
    # - https://www.microsoft.com/en-us/research/publication/2016/11/Digital-Signal-Processing.pdf
    # - http://research.microsoft.com/en-us/um/people/zhang/papers/tr03-39.pdf
    # - https://andrewkay.name/blog/post/aspect-ratio-of-a-rectangle-in-perspective/

    # Step 1: Get image center, will be used as origin
    h, w = image.shape[:2]
    origin = (w * .5, h * .5)

    # Step 2: Homeneous points coords from image origin
    # /!\ CAREFUL : points need to be in zig-zag order (A, B, D, C)
    p1 = numpy.array([*(corners[0] - origin), 1.])
    p2 = numpy.array([*(corners[1] - origin), 1.])
    p3 = numpy.array([*(corners[3] - origin), 1.])
    p4 = numpy.array([*(corners[2] - origin), 1.])

    # Step 3: Zhengyou Zhang p.10 : equations (11) & (12)
    k2 = numpy.dot(numpy.cross(p1, p4), p3) / numpy.dot(numpy.cross(p2, p4), p3)
    k3 = numpy.dot(numpy.cross(p1, p4), p2) / numpy.dot(numpy.cross(p3, p4), p2)

    # Step 4: Compute the focal length
    f = 0.
    f_sq = -((k3 * p3[1] - p1[1]) * (k2 * p2[1] - p1[1]) + \
             (k3 * p3[0] - p1[0]) * (k2 * p2[0] - p1[0]) ) / ((k3 - 1) * (k2 - 1))
    if f_sq > 0.:
        f = numpy.sqrt(f_sq)
    # If l_sq <= 0, λ cannot be computed, two sides of the rectangle's image are parallel
    # Either Uz and/or Vz is equal zero, so we leave l = 0

    # Step 5: Computing U & V vectors, BUT the z value of these vectors are in the form: z / f
    # Where f is the focal length
    u = (k2 * p2) - p1
    v = (k3 * p3) - p1

    # Step 6: Get length of U & V
    len_u = numpy.linalg.norm([u[0], u[1], (u[2] * f)])
    len_v = numpy.linalg.norm([v[0], v[1], (v[2] * f)])

    return (len_v / len_u)

def compute_paper_size(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32])-> tuple[int, int]:
    # Vectors of the side of the contour (clockwise)
    side_top_vec = corners[1] - corners[0]
    side_rgh_vec = corners[2] - corners[1]
    side_btm_vec = corners[2] - corners[3]
    side_lft_vec = corners[3] - corners[0]

    # Step 1: Compute average width & height of the paper sheet
    paper_avg_width = 0.5 * (numpy.linalg.norm(side_top_vec) + numpy.linalg.norm(side_btm_vec))
    paper_avg_height = 0.5 * (numpy.linalg.norm(side_lft_vec) + numpy.linalg.norm(side_rgh_vec))

    # Step 2: If deformation is negligable avoid computation and return the average dimensions
    #         Checking if the opposite sides are parallel
    if math.isclose((side_top_vec[0] * side_btm_vec[1]), (side_top_vec[1] * side_btm_vec[0]), abs_tol=PAPER_DEFORMATION_TOLERANCE) and \
        math.isclose((side_lft_vec[0] * side_rgh_vec[1]), (side_lft_vec[1] * side_rgh_vec[0]), abs_tol=PAPER_DEFORMATION_TOLERANCE):
        return (round(paper_avg_width), round(paper_avg_height))

    # Step 3: Compute aspect ratio
    aspect_ratio = compute_aspect_ratio(image, corners)

    if aspect_ratio == 0.:
        # The ratio could not be computed, use a fallback
        rect = cv.minAreaRect(corners)
        return (rect.size.width, rect.size.height)

    return (round(paper_avg_width), round(paper_avg_width * aspect_ratio))

def extract_paper_sheet(image: NDArray[numpy.uint8], corners: NDArray[numpy.float32])-> NDArray[numpy.uint8]:
    # Step 1: Compute size (width & height) of the paper sheet
    (width, height) = compute_paper_size(image, corners)

    # Step 2: Create the destination image size array
    dim_dest_image = numpy.array([[0., 0.], [(width - 1.), 0.], [(width - 1.), (height - 1.)], [0., (height - 1.)]])

    # Step 3: Compute the perspective deformation matrix
    #         /!\ inputs need to be numpy array in float32
    M = cv.getPerspectiveTransform(corners.astype(numpy.float32), dim_dest_image.astype(numpy.float32))

    # Step 4: Extract and unwrap/straighten the paper sheet
    paper_sheet = cv.warpPerspective(image, M, (int(width), int(height)), borderMode=cv.BORDER_CONSTANT, borderValue=(255, 255, 255))

    return paper_sheet

def levels_adjustment(image: NDArray[numpy.uint8], darks_val: list | int, lghts_val: list | int, gamma_val: float, min_level: list | int, max_level: list | int)-> NDArray[numpy.uint8]:
    in_darks = numpy.array([darks_val] if isinstance(darks_val, int) else darks_val, dtype=numpy.float32)
    in_lghts = numpy.array([lghts_val] if isinstance(lghts_val, int) else lghts_val, dtype=numpy.float32)
    in_gamma = numpy.array([gamma_val], dtype=numpy.float32)
    out_lmin = numpy.array([min_level] if isinstance(min_level, int) else min_level, dtype=numpy.float32)
    out_lmax = numpy.array([max_level] if isinstance(max_level, int) else max_level, dtype=numpy.float32)

    corrected = numpy.clip((image - in_darks) / (in_lghts - in_darks), 0, 255)
    corrected = (corrected ** (1. / in_gamma)) * (out_lmax - out_lmin) + out_lmin
    corrected = numpy.clip(corrected, 0, 255).astype(numpy.uint8)

    return corrected

def auto_levels_adjustment(image: NDArray[numpy.uint8], is_document: bool, clip_hist_percent: float = 2.0)-> NDArray[numpy.uint8]:
    # Step 1: Convert image to grayscale (Y of YUV color mode) if needed
    gray, _ = split_grayscale(image)

    # Step 3: Compute the normalized histogram
    hist = cv.calcHist(gray, [0], None, [256], (0, 256))
    hist = (hist / hist.sum()) * 100.0
    hist = hist.flatten()

    # Step 4: Find the levels darks index
    darks_val = 0
    percent = 0.0
    for index, curr_hist_val in enumerate(hist):
        percent += curr_hist_val
        darks_val = index
        if percent >= clip_hist_percent:
            break

    # Step 5: Find the levels lights index
    lghts_val = 0
    percent = 0.0
    for index in range(255, -1, -1):
        percent += hist[index]
        lghts_val = index
        if percent >= clip_hist_percent:
            break

    # Step 6: If it's a document we take the middle of lights part distribution
    #         A "better" solution should be to find the peak (limit extrema) of the lights part [190-255]
    if is_document:
        percent = 0.
        for index in range(190, lghts_val):
            percent += hist[index]
        limit = percent * .5
        percent = 0.

        for index in range(180, lghts_val):
            percent += hist[index]
            if percent >= limit:
                lghts_val = index
                break

    image = levels_adjustment(image, darks_val, lghts_val, 1.2, 0, 255)

    return image

def color_blend(gray_val: numpy.uint8, binary_val: numpy.uint8)-> numpy.uint8:
    # Slow but better in term of merged result image (not losing details), 1.2 is a constante value between [1.0, 3.0]
    weight = min(1.0, (1.2 * (gray_val / 255)) ** 2) if binary_val > 0 else 0.
    return int(((1.0 - weight) * gray_val) + (weight * binary_val))

def improve_paper_quality(paper: NDArray[numpy.uint8], paper_index: int, is_document: bool = True, max_quality: bool = False)-> NDArray[numpy.uint8]:
    # Step 1: Adjuste Levels
    corrected = auto_levels_adjustment(paper, is_document)
    if DEBUG:
        save_to_results_folder(corrected, 'Paper-{:03d}_DEBUG_08_Levels-Correction'.format(paper_index))

    if not is_document:
        # Skip document specific improvements
        return corrected

    # Step 2: Correct the Luminosity / Luma / Lightness
    gray, YUV = split_grayscale(corrected)
    height, width = gray.shape[:2]

    # Step 2.B: Create a binary image, white means full luminosity (255)
    # Blocksize is 1/10 of the smallest of the two dimensions rounded to the nearest odd number
    blocksize = int(math.ceil(min(width, height) * .05) * 2 + 1)
    c_const = round(blocksize * .1)# 1/10 of blocksize
    layer = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, blocksize, c_const)
    if DEBUG:
        save_to_results_folder(layer, 'Paper-{:03d}_DEBUG_09_Luma-Mask'.format(paper_index))

    # Step 2.C: Blend the binary image (layer) with the Luma (gray)
    if max_quality:
        vfunc = numpy.frompyfunc(color_blend, 2, 1)
        gray = vfunc(gray, layer).astype(numpy.uint8)
    else:
        gray = cv.addWeighted(gray, 1.0, layer, WHITE_CORRECTION_FACTOR, 0.)

    # Step 2.D: Set Luma (gray) as Y channel of YUV image
    YUV[:, :, 0] = gray

    return cv.cvtColor(YUV, cv.COLOR_YUV2BGR)

def main(images_paths: list[str])-> bool:
    """Entry point"""

    # Step 1: Create the result folder
    folder_created = create_results_folder()
    if not folder_created:
        return False

    # Step 2: Create Straightened versions of images
    paper_index = 1
    for image_path in images_paths:
        # Step 2.A: Ensure the path exists and is a valid file
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            LOG.error('Path "{}": Doesn\'t exist or isn\'t a file.'.format(path))
            continue

        # Step 2.B: Try reading / loading the image
        image = cv.imread(str(path.resolve()))
        if image is None:
            LOG.error('Path "{}": Cannot read the image.'.format(path))
            continue

        # Step 2.C: Retrieving the contour of the paper sheet if one is detected.
        #           Passing a copy of the image to be able to apply modifications
        #           like grayscale convertion or resizing.
        contour = retrieve_contour(image.copy(), paper_index)
        if contour is None:
            LOG.warning('Path "{}": Not able to find paper sheet in the image.'.format(path))
            continue

        # Step 2.D: Convert the contour to an array of points (a.k.a corners) sorted in clockwise order
        corners = get_corners_from_coutour(contour)

        # Step 2.E: Extract the paper sheet from the image and Straighten it
        paper = extract_paper_sheet(image, corners)
        if DEBUG:
            save_to_results_folder(paper, 'Paper-{:03d}_DEBUG_07_Extracted'.format(paper_index))

        paper = improve_paper_quality(paper, paper_index)

        # Step 2.F: Save the paper to the result folder
        filename = "Paper-{:03d}".format(paper_index)
        paper_index += 1
        save_to_results_folder(paper, filename)

    # Step 3: Delete the created results folder if empty
    if len(os.listdir(PATH_DIR_RESULTS)) == 0:
        delete_results_folder()

    return True

if __name__ == '__main__':
    import time
    import argparse
    parser = argparse.ArgumentParser(prog=SCRIPT_NAME, description='{} v{}, Detect Sheet of Paper, Extract & Straighten it.'.format(SCRIPT_NAME, VERSION))
    parser.add_argument('-v', '--version', action='version', version='%(prog)s '+ VERSION)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-i', '--images-paths', nargs='+', required=True, type=str, action='extend', help='disk path(s) to the image(s)')

    arguments = parser.parse_args()
    DEBUG = arguments.debug

    start_time = time.time()
    main(arguments.images_paths)
    LOG.info("Execution time: {} seconds.".format((time.time() - start_time)))
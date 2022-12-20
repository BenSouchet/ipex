import sys
import math
import shutil
import logging
import numpy
import cv2 as cv
from pathlib import Path
from datetime import datetime

SCRIPT_NAME = 'IPEX : Image Paper EXtractor'
VERSION = '0.0.3'

RESULT_IMAGE_EXT = '.png' # Can be any type handled by OpenCV, see documentation for valid values.
DETECTION_IMAGE_MAX_DIM = 1024 # In pixels, if the lagest dimension (width or height) of the input image is
#                                bigger than this value the image will be downscale ONLY for paper detect calculations.
#                                Smaller value mean faster computation but less accuracy.
KERNEL_ERODE_SIZE = 3 # Size in pixels of the "brush" for the erode operation, value need to be odd (3, 5, 7, ...)
SIMPLIFIED_CONTOUR_MAX_COEF = 0.15 # Maximum ratio of simplification allowed for the contour point reduction (e.g. simplify_contour function)
PAPER_DEFORMATION_TOLERANCE = 0.01 # Above this value a complexe method will be used to compute paper aspect ratio

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

def init_logger():
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

def delete_results_folder():
    if PATH_DIR_RESULTS and PATH_DIR_RESULTS.exists():
        try:
            shutil.rmtree(PATH_DIR_RESULTS)
        except Exception as e:
            LOG.error('Failed to delete the empty results folder. Reason: {}'.format(e))
            return False
    return True

def create_results_folder():
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

def save_to_results_folder(paper, filename):
    cv.imwrite(str(PATH_DIR_RESULTS.joinpath(filename)), paper)

def downscale_image(image):
    factor = 1.0
    height, width = image.shape[:2]

    # Step 1: If image doesn't need resize do nothing
    if height <= DETECTION_IMAGE_MAX_DIM and width <= DETECTION_IMAGE_MAX_DIM:
        return 1.0, image

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
    return factor, cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def auto_brightness_and_contrast(image, clip_hist_percent=1):
    gray = image
    # Step 1: Convert to grayscale if not already
    if len(image.shape) >= 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

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

def convert_to_binary_image(image):
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
    save_to_results_folder(layer_1, '05_A.png')### DEBUG

    # Step 3: Create the second layer using adaptive thresholding method
    layer_2 = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 17, 3)
    # Step 3.B: Applying a morphological closing to avoid discontinuated shapes
    layer_2 = cv.morphologyEx(layer_2, cv.MORPH_CLOSE, kernel_morph)
    save_to_results_folder(layer_2, '05_B.png')### DEBUG

    # Step 4: Merging the two layer by doing a substraction
    image = layer_1 - layer_2
    # Step 4.B: Clamping values with a threshold operation, required to ensure it's a valid binary image
    image = cv.threshold(image, 127, 255, cv.THRESH_BINARY)[1]

    # Step 5: Erode to ensure shapes are properly separated
    image = cv.erode(image, kernel_erode)

    # Step 6: Last pass to remove small blobs (optional)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel_morph)

    return image

def scale_contour_from_centroid(contour, scale):
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
    contour = contour.astype(numpy.int32)

    return contour

def simplify_contour_compute_weight(contour, index):
    p1 = contour[(index-1)%contour.shape[0]][0]
    p2 = contour[index][0]
    p3 = contour[(index+1)%contour.shape[0]][0]
    return (0.5 * abs((p1[0] * (p2[1] - p3[1])) + (p2[0] * (p3[1] - p1[1])) + (p3[0] * (p1[1] - p2[1]))))

def simplify_contour(contour, nbr_ptr_limit=4):
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

def find_paper_contour_from_binary_image(image):
    # Step 1: Find contours of shapes in the binary image
    contours = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[-2]

    # Step 2: Sort and choose the largest contour
    contours = sorted(contours, key = cv.contourArea)
    contour = contours[-1]

    # Step 3: Getting the shape from the contour
    arclen = cv.arcLength(contour, True)
    contour = cv.approxPolyDP(contour, 0.02 * arclen, True)
    if len(contour) < 4 or not cv.isContourConvex(contour):
        # The best shape candidate seems to not be a rectangle
        return None
    
    # Step 3.B: Try to simplify the contour to 4 points
    if len(contour) > 4:
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

def retrieve_contour(image):
    original = image.copy()### DEBUG
    downscale_factor2, original = downscale_image(original)### DEBUG

    # Step 1: Convert image to grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    save_to_results_folder(image, '01_Grayscale.png')### DEBUG

    # Step 2: Downscale the image if necessary, save the factor
    downscale_factor, image = downscale_image(image)
    save_to_results_folder(image, '02_Downscale.png')### DEBUG

    # Step 3: Median blur the image (a.k.a Noise Median Reduction),
    # this help removing the unecessary details while conserving the edges
    aperture_linear_size = 9
    image = cv.medianBlur(image, aperture_linear_size)
    save_to_results_folder(image, '03_MedianBlur.png')### DEBUG

    # Step 4: Correct the brightness and contrast (optional)
    image = auto_brightness_and_contrast(image, 5)
    save_to_results_folder(image, '04_Contrast.png')### DEBUG

    # Step 5: Convert to binary map (threshold & morphological transformations)
    image = convert_to_binary_image(image)
    save_to_results_folder(image, '05_Treshold.png')### DEBUG

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

    cv.drawContours(original, [contour], -1, (0, 0, 255), 1, cv.LINE_AA)### DEBUG
    save_to_results_folder(original, '06_Contour.png')### DEBUG

    # Step 7: Very important! Apply the downscale factor to scale up the contour to the correct size
    contour = (contour * (1.0 / downscale_factor))

    return contour

def get_corners_from_coutour(contour):
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

    # Step 3; Convert array to int
    corners = numpy.rint(corners).astype(int)

    return corners

def normalize_vector(vector):
    length = numpy.linalg.norm(vector)
    if length == 0:
       return vector
    return vector / length

def compute_aspect_ratio(image, corners):
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

    # Step 6: Get normalized U & V
    u = numpy.linalg.norm([u[0], u[1], (u[2] * f)])
    v = numpy.linalg.norm([v[0], v[1], (v[2] * f)])

    return (v / u)

def compute_paper_size(image, corners):
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
        return (paper_avg_width, paper_avg_height)

    # Step 3: Compute aspect ratio
    aspect_ratio = compute_aspect_ratio(image, corners)

    if aspect_ratio == 0.:
        # The ratio could not be computed, use a fallback
        rect = cv.minAreaRect(corners)
        return (rect.size.width, rect.size.height)

    return (paper_avg_width, paper_avg_width * aspect_ratio)

def extract_paper_sheet(image, corners):
    # Step 1: Compute size (width & height) of the paper sheet
    paper_size = compute_paper_size(image, corners)

    # Step 3: Create destination image

    # Step 4: Compute the perspective deformation matrix

    # Step 5: Unwrap / straighten the paper and save it to the destination image
    return image

def main(images_paths):
    """Entry point"""

    # Step 1: Create the result folder
    folder_created = create_results_folder()
    if not folder_created:
        return False

    result_folder_empty = False#debug

    # Step 2: Create Straightened versions of images
    paper_index = 0
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
        contour = retrieve_contour(image.copy())
        if contour is None:
            LOG.warning('Path "{}": Not able to find paper sheet in the image.'.format(path))
            continue

        # Step 2.D: Convert the contour to an array of points (a.k.a corners) sorted in clockwise order
        corners = get_corners_from_coutour(contour)

        # Step 2.E: Extract the paper sheet from the image and Straighten it
        paper = extract_paper_sheet(image, corners)

        # Step 2.F: Save the paper to the result folder
        filename = "paper_{:02d}{}".format(paper_index, RESULT_IMAGE_EXT)
        paper_index += 1
        save_to_results_folder(paper, filename)

    # Step 3: Delete the created results folder if empty
    if result_folder_empty:
        delete_results_folder()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog=SCRIPT_NAME, description='{} v{}, Detect Sheet of Paper, Extract & Straighten it.'.format(SCRIPT_NAME, VERSION))
    parser.add_argument('-v', '--version', action='version', version='%(prog)s '+ VERSION)
    parser.add_argument('-i', '--images-paths', nargs='+', required=True, type=str, action='extend', help='disk path(s) to the image(s)')

    arguments = parser.parse_args()

    main(arguments.images_paths)
""" 
This module contains functions for processing a frame
"""

import logging
import cv2
import numpy as np

from src.opencv_auto.utility import (
    show_image,
    length_of_line_segment,
    make_points,
    display_lines,
)

logger = logging.getLogger(__name__)


def detect_edges(frame):
    """
    Detect edges using Canny Edge Detection algorithm
    """

    # filter for blue lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    show_image("hsv", hsv)

    # mask the image to isolate only blue colors
    lower_blue = np.array([60, 40, 40])
    upper_blue = np.array([150, 190, 190])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    show_image("blue mask", mask)

    # detect edges
    edges = cv2.Canny(mask, 200, 400)

    return edges


def region_of_interest(canny):
    """
    Focus on a region of interest
    """
    height, width = canny.shape
    mask = np.zeros_like(canny)

    # only focus bottom half of the screen
    polygon = np.array(
        [
            [
                (0, height * 1 / 2),
                (width, height * 1 / 2),
                (width, height),
                (0, height),
            ]
        ],
        np.int32,
    )

    # fill the polygon with white
    cv2.fillPoly(mask, polygon, 255)
    show_image("mask", mask)

    # mask area of interest
    masked_image = cv2.bitwise_and(canny, mask)

    return masked_image


def detect_line_segments(masked_image):
    """
    Detect line segments using Probabilistic Hough Transform
    """
    precision = 1  # precision in pixel
    angle = np.pi / 180  # degree in radian
    min_threshold = 20  # minimal of votes #was 10

    # detect line segments
    line_segments = cv2.HoughLinesP(
        masked_image,
        precision,
        angle,
        min_threshold,
        np.array([]),
        minLineLength=8,
        maxLineGap=4,
    )

    if line_segments is not None:
        for line_segment in line_segments:
            logger.debug("detected line_segment:")
            logger.debug(
                f"{line_segment} of length {length_of_line_segment(line_segment[0])}"
            )

    return line_segments


def average_slope_intercept(frame, line_segments):
    """
    This function combines line segments into one or two lane lines
    If all line slopes are < 0: then we only have detected left lane
    If all line slopes are > 0: then we only have detected right lane
    """
    lane_lines = []
    if line_segments is None:
        logger.info("No line_segment segments detected")
        return lane_lines

    _, width, _ = frame.shape
    left_fit = []
    right_fit = []

    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary

    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                # logger.info(
                #     f"skipping vertical line segment (slope=inf): {line_segment}"
                # )
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))

    right_fit_average = np.average(right_fit, axis=0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))

    logger.debug(f"lane lines: {lane_lines}")

    return lane_lines


def detect_lane(frame, show_image_windows=False):
    """
    Detect lane lines in a frame
    """
    logger.debug("detecting lane lines...")

    edges = detect_edges(frame)
    if show_image_windows:
        show_image("edges", edges)

    cropped_edges = region_of_interest(edges)
    if show_image_windows:
        show_image("edges cropped", cropped_edges)

    line_segments = detect_line_segments(cropped_edges)
    line_segment_image = display_lines(frame, line_segments)
    if show_image_windows:
        show_image("line segments", line_segment_image)

    lane_lines = average_slope_intercept(frame, line_segments)
    lane_lines_image = display_lines(frame, lane_lines)
    if show_image_windows:
        show_image("lane lines", lane_lines_image)

    return lane_lines, lane_lines_image

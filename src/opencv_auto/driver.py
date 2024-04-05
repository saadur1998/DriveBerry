""" 
This file contains the driver for the lane detection using opencv.
"""

import logging

from .frame_processing import detect_lane
from .kinematics import compute_steering_angle, stabilize_steering_angle
from .utility import display_heading_line, show_image

logger = logging.getLogger(__name__)


class AutoDrive(object):
    """
    Base class for a self-driving car
    """

    def __init__(self, car=None):
        """
        car: the car that should be controlled by this object
        """
        logger.info("Creating an instance of AutoDrive")
        self.car = car
        self.curr_steering_angle = 90

    def follow_lane(self, frame):
        """
        Main entry point of the lane follower
        """

        show_image("orig", frame)

        lane_lines, frame = detect_lane(frame, show_image_windows=False)
        final_frame = self.steer(frame, lane_lines)

        return final_frame

    def steer(self, frame, lane_lines):
        """
        Steer the car based on lane line coordinate
        """
        logger.debug("steering...")
        if len(lane_lines) == 0:
            logger.error("No lane lines detected, nothing to do.")
            return frame

        new_steering_angle = compute_steering_angle(frame, lane_lines)
        self.curr_steering_angle = stabilize_steering_angle(
            self.curr_steering_angle, new_steering_angle, len(lane_lines)
        )

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
        show_image("heading", curr_heading_image)

        return curr_heading_image

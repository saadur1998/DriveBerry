""" 
This module contains the main class for the lane follower.
"""

import logging
import math

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from src.cnn_driving.utility import img_preprocess, display_heading_line, show_image

logger = logging.getLogger(__name__)


class CNNDrive(object):
    """
    Base class for a self-driving car
    """

    def __init__(
        self, car=None, model_path="src/cnn_driving/model/lane_follower_cnn.keras"
    ):
        logger.info("Creating a CNNDrive instance...")

        self.car = car
        self.curr_steering_angle = 90
        self.model = load_model(model_path)

    def compute_steering_angle(self, frame):
        """
        Compute the steering angle based on the input frame
        """
        preprocessed = img_preprocess(frame)
        x = np.asarray([preprocessed])
        steering_angle = self.model.predict(x)[0]

        logger.debug("new steering angle: %s" % steering_angle)
        return int(steering_angle + 0.5)

    def follow_lane(self, frame):
        """
        Main entry point of the lane follower
        """

        show_image("orig", frame)

        self.curr_steering_angle = self.compute_steering_angle(frame)
        logger.debug("curr_steering_angle = %d" % self.curr_steering_angle)

        if self.car is not None:
            self.car.front_wheels.turn(self.curr_steering_angle)
        final_frame = display_heading_line(frame, self.curr_steering_angle)

        return final_frame

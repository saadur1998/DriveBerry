""" 
    This module is responsible for detecting objects on the road
"""
import logging
import time

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

import cv2

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DetectionModel(object):
    """
    This class is responsible for detecting objects on the road
    """

    def __init__(
        self,
        car=None,
        speed_limit=40,
        model_path="src/object_detection/artifacts/road_signs_quantized_edgetpu.tflite",
        label_path="src/object_detection/artifacts/labels.txt",
    ):
        """
        Initialize the DetectionModel
        """
        logger.info("Initializing DetectionModel")

        # set variables
        self.car = car
        self.speed_limit = speed_limit
        self.width = 640
        self.height = 480

        # model related config
        self.model_path = model_path
        self.label_path = label_path
        self.labels = self.load_labels(self.label_path)
        self.interpreter = make_interpreter(self.model_path)
        self.interpreter.allocate_tensors()

        # stop sign related config
        self.stopped = False
        self.stop_sign_count = 0

    def load_labels(self, path):
        """
        Load labels from text file.
        """
        with open(path, "r") as file:
            return {i: line.strip() for i, line in enumerate(file.readlines())}

    def detect_objects(self, frame):
        """
        Detect objects on the road
        """

        # Load labels
        labels = self.labels

        # Read and preprocess an image.
        _, scale = common.set_resized_input(
            self.interpreter, frame.shape[:2], lambda size: cv2.resize(frame, size)
        )

        start = time.perf_counter()

        # Run inference
        self.interpreter.invoke()
        inference_time = time.perf_counter() - start

        # Get detection results
        results = detect.get_objects(
            self.interpreter, score_threshold=0.65, image_scale=scale
        )

        logger.debug("%.2f ms" % (inference_time * 1000))

        # Print labels of detected objects
        if results and len(results) > 0:
            for obj in results:
                label = f"{labels[obj.id]} {obj.score:.2f}"
                logger.info(label)

        else:
            logger.debug("No objects detected")

        return results, frame

    def is_close_by(self, obj, frame_height, min_height_pct=-0.107):
        """
        Check if object is close by (Not used yet)
        """
        bbox = obj.bbox
        obj_height = bbox.ymin - bbox.ymax
        print(obj_height / frame_height)
        return obj_height / frame_height > min_height_pct

    def control_car(self, objects):
        """
        Control the car based on detected objects
        """
        logger.debug("Controlling car")
        if len(objects) == 0:
            logger.debug("No objects detected, continue driving")
        else:
            for obj in objects:
                label = self.labels[obj.id]

                # handle stop sign
                if label == "stop sign":
                    self.stop_sign_count += 1
                    logger.info("Detected Stop Sign")
                    if self.stop_sign_count == 2 and self.stopped is False:
                        logger.info("Stopping car for 3 seconds")

                        # stop and start the car
                        self.car.back_wheels.speed = 0
                        time.sleep(3)
                        self.car.back_wheels.speed = self.speed_limit
                        self.stopped = True
                        self.stop_sign_count = 0

    def process_objects_on_road(self, frame):
        """
        Main entry point of the Road Object Handler
        """
        logger.debug("Processing objects")
        objects, final_frame = self.detect_objects(frame)
        self.control_car(objects)
        logger.debug("Processing objects END")

        return final_frame

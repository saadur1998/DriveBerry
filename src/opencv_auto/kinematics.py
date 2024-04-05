""" 
    Kinematics of the car based on the detected lane lines
"""
import logging
import math

logger = logging.getLogger(__name__)


def compute_steering_angle(frame, lane_lines):
    """
    Find the steering angle based on lane line coordinate
    """
    if len(lane_lines) == 0:
        logger.info("No lane lines detected, do nothing")
        return -90

    height, width, _ = frame.shape
    if len(lane_lines) == 1:
        logger.debug(f"Only detected one lane line, just follow it. {lane_lines[0]}")
        x1, _, x2, _ = lane_lines[0][0]
        x_offset = x2 - x1
    else:
        _, _, left_x2, _ = lane_lines[0][0]
        _, _, right_x2, _ = lane_lines[1][0]
        camera_mid_offset_percent = -0.035
        mid = int(width / 2 * (1 + camera_mid_offset_percent))
        x_offset = (left_x2 + right_x2) / 2 - mid

    # find the steering angle, which is angle between navigation direction to end of center line
    y_offset = int(height / 2)

    angle_to_mid_radian = math.atan(x_offset / y_offset)
    angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
    steering_angle = angle_to_mid_deg + 90

    logger.debug(f"new steering angle: {steering_angle}")
    return steering_angle


def stabilize_steering_angle(
    curr_steering_angle,
    new_steering_angle,
    num_of_lane_lines,
    max_angle_deviation_two_lines=5,
    max_angle_deviation_one_lane=1,
):
    """
    Using last steering angle to stabilize the steering angle
    This can be improved to use last N angles, etc
    if new angle is too different from current angle, only turn by max_angle_deviation degrees
    """
    if num_of_lane_lines == 2:
        max_angle_deviation = max_angle_deviation_two_lines
    else:
        max_angle_deviation = max_angle_deviation_one_lane

    angle_deviation = new_steering_angle - curr_steering_angle
    if abs(angle_deviation) > max_angle_deviation:
        stabilized_steering_angle = int(
            curr_steering_angle
            + max_angle_deviation * angle_deviation / abs(angle_deviation)
        )
    else:
        stabilized_steering_angle = new_steering_angle
    logger.debug(
        f"Proposed angle: {new_steering_angle}; stabilized angle: {stabilized_steering_angle}"
    )
    return stabilized_steering_angle

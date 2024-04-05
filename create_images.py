""" 
This script is used to create images and steering angle from a video file.
"""
import os
import sys
import cv2


from src.opencv_auto.driver import AutoDrive


def save_image_and_steering_angle(filename):
    """
    Save images and steering angle from a video file
    """
    lane_follower = AutoDrive()
    cap = cv2.VideoCapture(f"{filename}.avi")

    try:
        # find the number of images already saved
        directory = os.fsencode("./data/images")
        print(f"found {len(os.listdir(directory))} images")
        i = len(os.listdir(directory)) + 5

        while cap.isOpened():
            _, frame = cap.read()
            lane_follower.follow_lane(frame)
            cv2.imwrite(
                f"./data/images/{i:03d}_{lane_follower.curr_steering_angle:03d}.png",
                frame,
            )
            i += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    save_image_and_steering_angle(sys.argv[1])

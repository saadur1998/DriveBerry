""" 
    File to test the object detection performance on Edge TPU
"""

import time

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.edgetpu import make_interpreter

import cv2


def load_labels(path):
    """
    Load labels from text file.
    """
    with open(path, "r") as file:
        return {i: line.strip() for i, line in enumerate(file.readlines())}


def draw_objects(image, results, labels):
    """
    Draw bounding boxes and labels for each detection in image.
    """
    for obj in results:
        # Draw the bounding box
        bbox = obj.bbox
        cv2.rectangle(
            image, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0, 255, 0), 2
        )

        # Draw label and score
        label = f"{labels[obj.id]} {obj.score:.2f}"
        print(f"Detected object: {label}")
        print(label)
        cv2.putText(
            image,
            label,
            (bbox.xmin, bbox.ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            1,
        )

    # Display the image
    cv2.imshow("Detected Objects", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def detect_objects(frame, interpreter, label_path):
    """
    Detect objects on the road
    """
    start = time.time()
    # Load the TFLite model and allocate tensors.

    # Load labels
    labels = load_labels(label_path)

    # Read and preprocess an image.
    # image = cv2.imread(image_path)
    _, scale = common.set_resized_input(
        interpreter, frame.shape[:2], lambda size: cv2.resize(frame, size)
    )

    # Run inference
    interpreter.invoke()

    # Get detection results
    results = detect.get_objects(interpreter, score_threshold=0.2, image_scale=scale)

    end = time.time()
    print(f"Time elapsed: {end-start}")
    print(f"FPS: {1/(end-start)}")
    print(f"Number of objects detected: {len(results)}")

    # Draw the results on the image
    draw_objects(frame, results, labels)


# Example usage
model_path = "src/object_detection/artifacts/road_signs_quantized_edgetpu.tflite"  # Path to the TFLite model file

# create model interpreter
interpreter = make_interpreter(model_path)
interpreter.allocate_tensors()

label_path = "src/object_detection/artifacts/labels.txt"  # Path to the labels file


# capture video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        detect_objects(frame, interpreter, label_path)
    else:
        break

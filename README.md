# DriveBerry: A Raspberry Pi Based Autonomous Vehicle

## Introduction

Welcome to driveberry, a project as part of the MSML650: Machine Learning for Robotics class, focused on developing an autonomous vehicle using a Raspberry Pi. This repository is your gateway to building a basic self-driving vehicle, leveraging the power and flexibility of Raspberry Pi.

## About the Project

Driveberry utilizes a Raspberry Pi to control and automate a vehicle. It uses OpenCV to build a lane following teacher network. A student CNN network is trained using the videos captured from the camera while using the teacher network to control the vehicle. To use the OpenCV based network, use the AutoDrive class and to use the CNN, use CNNDrive class.

## Key Features

- Utilizes the Raspberry Pi and an Edge TPU to build a basic autonomous vehicle that can follow lanes and detect stop signs
- Fully coded in Python.
- Uses OpenCV to build a lane following teacher network.
- A CNN is trained using the videos captured from the camera while using the teacher network to control the vehicle.
- Object detection network is a quantized tflite model pretrained and taken from the DeepPiCar repo that can detect stop signs. Runs on the Edge TPU with 30-40FPS as opposed to 1-2FPS on the CPU.

## Requirements

- **Raspberry Pi**: Model 3B or higher recommended for optimal performance.
- **Hardware Components**:
  - Edge TPU.
  - SunFounder PiCar V Kit.
  - A 170 degree camera.
  - 2 18650 batteries.
- **Software**:
  - Python 3.x (Not compatible with python 3.10+, conda environments recommended).
  - OpenCV, Tensorflow, TFLite, PyCoral, Numpy, Pandas, Picar.

## Installation and Setup

1. Assemble the Raspberry Pi and connect all hardware components.
2. Install python and dependencies in the virtual environment.
3. Clone this repository to your Raspberry Pi.
4. Run the main script using Python to start the vehicle's autonomous operations.

## Usage

- Run the main script using Python to start the vehicle's autonomous operations.
- The vehicle can be operated using teacher or the student network which can be selected through the script.
- For troubleshooting and detailed operational procedures, contact Aditya at apatkar@umd.edu

## Contributing

We encourage contributions to the driveberry project. You can contribute by:

- Reporting bugs and submitting issue reports.
- Proposing new features or enhancements.
- Making pull requests for bug fixes or new functionalities.

## License

This project is released under the MIT License. See the LICENSE file for more details.

## Acknowledgements

A special thanks to professor Jerry Wu and the TA Aman Sharma at `University of Maryland` for their guidance and support throughout the project.

## Contact Information

For further inquiries or to report issues, please contact Aditya at apatkar@umd.edu and Saad at saadr98@umd.edu.

Feel free to explore, learn, and contribute to the driveberry project.

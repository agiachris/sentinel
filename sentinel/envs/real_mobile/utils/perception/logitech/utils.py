# Author: Jimmy Wu
# Date: September 2022

import os
import json
import sys
import cv2 as cv
import math


def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


################################################################################
# Camera


def get_video_cap(serial, frame_width, frame_height, new_camera=False):
    if sys.platform == "darwin":
        return cv.VideoCapture(0)
    cap = cv.VideoCapture(
        f"/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e_{serial}-video-index0"
    )
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
    cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Gives much better latency
    assert cap.get(cv.CAP_PROP_FRAME_WIDTH) == frame_width
    assert cap.get(cv.CAP_PROP_FRAME_HEIGHT) == frame_height
    assert cap.get(cv.CAP_PROP_BUFFERSIZE) == 1

    # White balance
    cap.set(cv.CAP_PROP_AUTO_WB, 0)
    cap.set(cv.CAP_PROP_TEMPERATURE, 3900)
    assert cap.get(cv.CAP_PROP_AUTO_WB) == 0
    assert cap.get(cv.CAP_PROP_TEMPERATURE) == 3900

    # Focus
    cap.set(cv.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv.CAP_PROP_FOCUS, 0)
    assert cap.get(cv.CAP_PROP_AUTOFOCUS) == 0
    assert cap.get(cv.CAP_PROP_FOCUS) == 0

    # Exposure
    cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = off, 3 = on
    assert cap.get(cv.CAP_PROP_AUTO_EXPOSURE) == 1

    # Fixed gain/exposure can be unreliable
    for _ in range(30):  # Read several frames to let exposure stabilize
        cap.read()
        cap.set(cv.CAP_PROP_GAIN, 50)
        cap.set(cv.CAP_PROP_EXPOSURE, 77)
    assert cap.get(cv.CAP_PROP_GAIN) == 50
    assert cap.get(cv.CAP_PROP_EXPOSURE) == 77

    return cap


def get_camera_params(serial):
    dir_name = os.path.dirname(os.path.realpath(__file__))
    camera_params_file_path = os.path.join(dir_name, "camera_params", f"{serial}.yml")
    assert os.path.isfile(camera_params_file_path)
    fs = cv.FileStorage(str(camera_params_file_path), cv.FILE_STORAGE_READ)
    image_width = int(fs.getNode("image_width").real())
    image_height = int(fs.getNode("image_height").real())
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return image_width, image_height, camera_matrix, dist_coeffs


def get_camera_alignment_params(serial):
    dir_name = os.path.dirname(os.path.realpath(__file__))
    params_path = os.path.join(dir_name, "camera_params", f"{serial}.json")
    assert os.path.isfile(params_path)
    with open(params_path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    return labels["camera_center"], labels["camera_corners"]


################################################################################
# Printouts


def get_paper_params(orientation="P"):
    width, height, margin = 8.5, 11, 0.5
    if orientation == "L":
        width, height = height, width
    ppi = 600
    mm_per_in = 25.4
    params = {}
    params["width_mm"] = mm_per_in * width
    params["height_mm"] = mm_per_in * height
    params["margin_mm"] = mm_per_in * margin
    params["mm_per_printed_pixel"] = mm_per_in / ppi
    return params

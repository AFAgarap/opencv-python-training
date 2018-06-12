from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__version__ = '1.0.0'
__author__ = 'Abien Fred Agarap'

import cv2
import numpy as np

cap = cv2.VideoCapture(-1)

def apply_invert(frame):
    return cv2.bitwise_not(frame)

def apply_sepia(frame, intensity=0.5):
    blue = 20
    green = 66
    red = 112
    frame = apply_color_overlay(frame, intensity=intensity, blue=blue, green=green, red=red)
    return frame

def verify_alpha_channel(frame):
    try:
        frame.shape[3]
    except IndexError:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
    return frame

def apply_color_overlay(frame, intensity=0.5, blue=0, green=0, red=0):
    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_channel = frame.shape
    sepia_bgra = (blue, green, red, 1)

    # create overlay based on sepia colors
    overlay = np.full((frame_height, frame_width, 4), sepia_bgra, dtype='uint8')

    # add sepia overlay on frame
    frame = cv2.addWeighted(overlay, intensity, frame, 1.0, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def apply_circle_focus_blur(frame, intensity=0.2):
    frame = verify_alpha_channel(frame)
    frame_height, frame_width, frame_channel = frame.shape
    y = int(frame_height / 2)
    x = int(frame_width / 2)
    radius = int(y / 2)
    center = (x, y)
    mask = np.zeros((frame_height, frame_width, 4), dtype='uint8')
    cv2.circle(mask, center, radius, (255, 255, 255), -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (21, 21), 11)
    blurred = cv2.GaussianBlur(frame, (21, 21), 11)
    blended = alpha_blend(frame, blurred, 255 - mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame

def alpha_blend(frame_1, frame_2, mask):
    alpha = mask / 255.0
    blended = cv2.convertScaleAbs(frame_1 * (1 - alpha) + (frame_2 * alpha))
    return blended

def apply_portrait_mode(frame):
    frame = verify_alpha_channel(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    blurred = cv2.GaussianBlur(frame, (21, 21), 0)

    blended = alpha_blend(frame, blurred, mask)
    frame = cv2.cvtColor(blended, cv2.COLOR_BGRA2BGR)
    return frame

while True:
    ret, frame = cap.read()

    invert = apply_invert(frame)
    sepia = apply_sepia(frame)
    redish_color = apply_color_overlay(frame, intensity=0.5, red=230, blue=10)
    circle_blurred = apply_circle_focus_blur(frame)
    portrait_mode = apply_portrait_mode(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('invert', invert)
    cv2.imshow('sepia', sepia)
    cv2.imshow('redish_color', redish_color)
    cv2.imshow('circle_blurred', circle_blurred)
    cv2.imshow('portrait_mode', portrait_mode)

    k = cv2.waitKey(1)

    if k == ord('q') or k == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
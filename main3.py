import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
hands_videos = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

sample_img = cv2.imread('265_E39A3_.jpg')

# Specify a size of the figure.
# plt.figure(figsize=[10, 10])

# Display the sample image, also convert BGR to RGB for display.
# plt.title("Sample Image")
# plt.axis('off')
# plt.imshow(sample_img[:, :, ::-1])
# plt.show()

# results = hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

# # Check if landmarks are found.
# if results.multi_hand_landmarks:
#
#     # Iterate over the found hands.
#     for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
#
#         print(f'HAND NUMBER: {hand_no + 1}')
#         print('-----------------------')
#
#         # Iterate two times as we only want to display first two landmarks of each hand.
#         for i in range(2):
#             # Display the found normalized landmarks.
#             print(f'{mp_hands.HandLandmark(i).name}:')
#             print(f'{hand_landmarks.landmark[mp_hands.HandLandmark(i).value]}')
#
# image_height, image_width, _ = sample_img.shape
#
# # Check if landmarks are found.
# if results.multi_hand_landmarks:
#
#     # Iterate over the found hands.
#     for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
#
#         print(f'HAND NUMBER: {hand_no + 1}')
#         print('-----------------------')
#
#         # Iterate two times as we only want to display first two landmark of each hand.
#         for i in range(2):
#             # Display the found landmarks after converting them into their original scale.
#             print(f'{mp_hands.HandLandmark(i).name}:')
#             print(f'x: {hand_landmarks.landmark[mp_hands.HandLandmark(i).value].x * image_width}')
#             print(f'y: {hand_landmarks.landmark[mp_hands.HandLandmark(i).value].y * image_height}')
#             print(f'z: {hand_landmarks.landmark[mp_hands.HandLandmark(i).value].z * image_width}\n')

# img_copy = sample_img.copy()

# # Check if landmarks are found.
# if results.multi_hand_landmarks:
#
#     # Iterate over the found hands.
#     for hand_no, hand_landmarks in enumerate(results.multi_hand_landmarks):
#         # Draw the hand landmarks on the copy of the sample image.
#         mp_drawing.draw_landmarks(image=img_copy, landmark_list=hand_landmarks,
#                                   connections=mp_hands.HAND_CONNECTIONS)
#
#     # Specify a size of the figure.
#     fig = plt.figure(figsize=[10, 10])
#
#     # Display the resultant image with the landmarks drawn, also convert BGR to RGB for display.
#     plt.title("Resultant Image")
#     plt.axis('off')
#     plt.imshow(img_copy[:, :, ::-1])
#     plt.show()


def detectHandsLandmarks(image, hands, display=True):
    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found.
    if results.multi_hand_landmarks:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the copy of the input image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS)

            # Check if the original input image and the output image are specified to be displayed.
    if display:
        print("display")
        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output")
        plt.axis('off')

    # Otherwise
    else:
        print("not displayed")
        # Return the output image and results of hands landmarks detection.
        return output_image, results

image = cv2.imread('landmark.png')
detectHandsLandmarks(image, hands, display=True)
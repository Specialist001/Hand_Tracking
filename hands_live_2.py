from operator import gt

import cv2
import time
import pygame
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

hands_video = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7,
                             min_tracking_confidence=0.4)

# Initialize the VideoCapture object to read from the webcam.
camera_video = cv2.VideoCapture(0)
# camera_video.set(3, 1280)
# camera_video.set(4, 960)

# Create named window for resizing purposes.
cv2.namedWindow('Hands Landmarks Detection', cv2.WINDOW_NORMAL)

# Initialize a variable to store the time of the previous frame.
time1 = 0


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
        # print("not displayed")
        # Return the output image and results of hands landmarks detection.
        return output_image, results


def customLandmarksAnnotation(image, landmark_dict):
    '''
    This function draws customized landmarks annotation utilizing the z-coordinate (depth) values of the hands.
    Args:
        image:         The image of the hands on which customized landmarks annotation of the hands needs to be drawn.
        landmark_dict: The dictionary that stores the hand(s) landmarks as different elements with keys as hand
                       types(i.e., left and right).
    Returns:
        output_image: The image of the hands with the customized annotation drawn.
        depth:        A dictionary that contains the average depth of all landmarks of the hand(s) in the image.
    '''

    # Create a copy of the input image to draw annotation on.
    output_image = image.copy()

    # Initialize a dictionary to store the average depth of all landmarks of hand(s).
    depth = {}

    # Initialize a list with the arrays of indexes of the landmarks that will make the required
    # line segments to draw on the hand.
    segments = [np.arange(0, 5), np.arange(5, 9), np.arange(9, 13), np.arange(13, 17), np.arange(17, 21),
                np.arange(5, 18, 4), np.array([0, 5]), np.array([0, 17])]

    # Iterate over the landmarks dictionary.
    for hand_type, hand_landmarks in landmark_dict.items():

        # Get all the z-coordinates (depth) of the landmarks of the hand.
        depth_values = np.array(hand_landmarks)[:, -1]

        # Calculate the average depth of the hand.
        average_depth = int(sum(depth_values) / len(depth_values))

        # Get all the x-coordinates of the landmarks of the hand.
        x_values = np.array(hand_landmarks)[:, 0]

        # Get all the y-coordinates of the landmarks of the hand.
        y_values = np.array(hand_landmarks)[:, 1]

        # Initialize a list to store the arrays of x and y coordinates of the line segments for the hand.
        line_segments = []

        # Iterate over the arrays of indexes of the landmarks that will make the required line segments.
        for segment_indexes in segments:
            # Get an array of a line segment coordinates of the hand.
            line_segment = np.array([[int(x_values[index]), int(y_values[index])] for index in segment_indexes])

            # Append the line segment coordinates into the list.
            line_segments.append(line_segment)

        # Check if the average depth of the hand is less than 0.
        if average_depth < 0:

            # Set the thickness of the line segments of the hand accordingly to the average depth.
            line_thickness = int(np.ceil(0.1 * abs(average_depth))) + 2

            # Set the thickness of the circles of the hand landmarks accordingly to the average depth.
            circle_thickness = int(np.ceil(0.1 * abs(average_depth))) + 3

        # Otherwise.
        else:

            # Set the thickness of the line segments of the hand to 2 (i.e. the minimum thickness we are specifying).
            line_thickness = 2

            # Set the thickness of the circles to 3 (i.e. the minimum thickness)
            circle_thickness = 3

        # Draw the line segments on the hand.
        cv2.polylines(output_image, line_segments, False, (100, 250, 55), line_thickness)

        # Write the average depth of the hand on the output image.
        cv2.putText(output_image, 'Depth: {}'.format(average_depth), (10, 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (20, 25, 255), 1,
                    cv2.LINE_AA)

        # Iterate over the x and y coordinates of the hand landmarks.
        for x, y in zip(x_values, y_values):
            # Draw a circle on the x and y coordinate of the hand.
            cv2.circle(output_image, (int(x), int(y)), circle_thickness, (55, 55, 250), -1)

        # Store the calculated average depth in the dictionary.
        depth[hand_type] = average_depth

    # Return the output image and the average depth dictionary of the hand(s).
    return output_image, depth


def getHandType(image, results, draw=True, display=True):
    # Create a copy of the input image to write hand type label on.
    output_image = image.copy()

    # Initialize a dictionary to store the classification info of both hands.
    hands_status = {'Right': False, 'Left': False, 'Right_index': None, 'Left_index': None}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand.
        hand_type = hand_info.classification[0].label

        # Update the status of the found hand.
        hands_status[hand_type] = True

        # Update the index of the found hand.
        hands_status[hand_type + '_index'] = hand_index

        # Check if the hand type label is specified to be written.
        if draw:
            # Write the hand type on the output image.
            cv2.putText(output_image, hand_type + ' Hand Detected', (10, (hand_index + 1) * 30), cv2.FONT_HERSHEY_PLAIN,
                        2, (0, 255, 0), 2)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the output image and the hands status dictionary that contains classification info.
        return output_image, hands_status


def drawBoundingBoxes(image, results, hand_status, padd_amount=10, draw=True, display=True):
    # Create a copy of the input image to draw bounding boxes on and write hands types labels.
    output_image = image.copy()

    # Initialize a dictionary to store both (left and right) hands landmarks as different elements.
    output_landmarks = {}

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Iterate over the found hands.
    for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):

        # Initialize a list to store the detected landmarks of the hand.
        landmarks = []

        # Iterate over the detected landmarks of the hand.
        for landmark in hand_landmarks.landmark:
            # Append the landmark into the list.
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))

        # Get all the x-coordinate values from the found landmarks of the hand.
        x_coordinates = np.array(landmarks)[:, 0]

        # Get all the y-coordinate values from the found landmarks of the hand.
        y_coordinates = np.array(landmarks)[:, 1]

        # Get the bounding box coordinates for the hand with the specified padding.
        x1 = int(np.min(x_coordinates) - padd_amount)
        y1 = int(np.min(y_coordinates) - padd_amount)
        x2 = int(np.max(x_coordinates) + padd_amount)
        y2 = int(np.max(y_coordinates) + padd_amount)

        # Initialize a variable to store the label of the hand.
        label = "Unknown"

        # Check if the hand we are iterating upon is the right one.
        if hand_status['Right_index'] == hand_index:

            # Update the label and store the landmarks of the hand in the dictionary.
            label = 'Right Hand'
            output_landmarks['Right'] = landmarks

        # Check if the hand we are iterating upon is the left one.
        elif hand_status['Left_index'] == hand_index:

            # Update the label and store the landmarks of the hand in the dictionary.
            label = 'Left Hand'
            output_landmarks['Left'] = landmarks

        # Check if the bounding box and the classified label is specified to be written.
        if draw:
            # Draw the bounding box around the hand on the output image.
            cv2.rectangle(output_image, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)

            # Write the classified label of the hand below the bounding box drawn.
            cv2.putText(output_image, label, (x1, y2 + 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (20, 255, 155), 1,
                        cv2.LINE_AA)

    # Check if the output image is specified to be displayed.
    if display:

        # Display the output image.
        plt.figure(figsize=[10, 10])
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the output image and the landmarks dictionary.
        return output_image, output_landmarks


# Iterate until the webcam is accessed successfully.
while camera_video.isOpened():

    # Read a frame.
    ok, frame = camera_video.read()

    # Check if frame is not read properly then continue to the next iteration to read the next frame.
    if not ok:
        continue

    frame = cv2.flip(frame, 1)

    # Flip the frame horizontally for natural (selfie-view) visualization.
    # frame = cv2.flip(frame, 1)

    # Perform Hands landmarks detection.
    annotated_frame, results = detectHandsLandmarks(frame, hands_video, display=False)

    if results.multi_hand_landmarks:
        # Perform hand(s) type (left or right) classification.
        _, hands_status = getHandType(frame.copy(), results, draw=False, display=False)

        # Get the landmarks dictionary that stores each hand landmarks as different elements.
        frame, landmark_dict = drawBoundingBoxes(frame, results, hands_status, draw=False, display=False)

        # Draw customized landmarks annotation ultilizing the z-coordinate (depth) values of the hand(s).
        custom_ann_frame, _ = customLandmarksAnnotation(frame, landmark_dict)

        # Stack the frame annotated using mediapipe with the customized one.
        final_output = np.hstack((annotated_frame, custom_ann_frame))
    else:
        # Stack the frame two time.
        print("not")
        final_output = np.hstack((frame, frame))

    # Display the frame.
    cv2.imshow('Hands Landmarks Detection', frame)

    # Wait for 1ms. If a a key is pressed, retreive the ASCII code of the key.
    k = cv2.waitKey(1)

    # Check if 'ESC' is pressed and break the loop.
    if (k == 27):
        break

# Release the VideoCapture Object and close the windows.
camera_video.release()
cv2.destroyAllWindows()

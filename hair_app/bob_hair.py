import dlib
import numpy as np
import cv2
from math import hypot

hair_image = cv2.imread('bob.png')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('C:\\Users\\Laravel\\sailoonai\\manish\\'
                                'djangoMedia\\newApp\\shape_predictor_68_face_landmarks.dat')
    
image = cv2.imread('test1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = detector(gray)
for face in faces:

    landmarks = predictor(gray, face)

    left_ear = (landmarks.part(0).x, landmarks.part(0).y)
    right_ear = (landmarks.part(16).x, landmarks.part(16).y)


    center_point = (landmarks.part(27).x + 12, landmarks.part(27).y - 40)
    hair_width = int(hypot(left_ear[0] - right_ear[0], left_ear[1] - right_ear[1])*1.7)
    hair_height = int(hair_width * 0.97)

    top_left = (int(center_point[0] - hair_width / 2), int(center_point[1] - hair_height / 2))
    # bottom_right = (int(center_point[0] + hair_width / 2), int(center_point[1] + hair_height / 2))

    hair = cv2.resize(hair_image, (hair_width, hair_height))
    hair_gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
    _, hair_mask = cv2.threshold(hair_gray, 1, 255, cv2.THRESH_BINARY_INV)

    hair_area = image[top_left[1]: top_left[1] + hair_height, top_left[0]: top_left[0] + hair_width]
    hair_area_no_hair = cv2.bitwise_or(hair_area, hair_area, mask=hair_mask)
    final_output = cv2.add(hair_area_no_hair, hair)

    image[top_left[1]: top_left[1] + hair_height, top_left[0]: top_left[0] + hair_width] = final_output

cv2.imshow('bob image', image)
cv2.waitKey(0)

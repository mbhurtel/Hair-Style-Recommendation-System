def process_red_hair(img):
    import cv2
    import dlib
    from math import hypot
    from keras.preprocessing import image
    import numpy as np

    hair_image = cv2.imread('red_emo.png')

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)
        left_point = (landmarks.part(0).x - 50, landmarks.part(0).y)
        right_point = (landmarks.part(16).x + 50, landmarks.part(16).y)
        center_point = (landmarks.part(27).x, landmarks.part(27).y - 58)

        hair_width = int(hypot(left_point[0] - right_point[0], left_point[1] - right_point[1]) * 1.2)
        hair_height = int(hair_width * 0.84)

        hair = cv2.resize(hair_image, (hair_width, hair_height))

        top_left = (int(center_point[0] - hair_width / 2), int(center_point[1] - hair_height / 2))

        hair_area = img[top_left[1]: top_left[1] + hair_height, top_left[0]: top_left[0] + hair_width]

        gray_hair = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
        _, hair_mask = cv2.threshold(gray_hair, 5, 255, cv2.THRESH_BINARY_INV)

        hair_area_no_hair = cv2.bitwise_or(hair_area, hair_area, mask=hair_mask)

        final = cv2.add(hair_area_no_hair, hair)

        img[top_left[1]: top_left[1] + hair_height, top_left[0]: top_left[0] + hair_width] = final
        # img.setflags(write=1)

    return(img)


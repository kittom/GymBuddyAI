import time

import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils # Get necessary utilities for drawing solutions

# cap = cv2.VideoCapture('PoseVideos/4.mp4')
cap = cv2.VideoCapture(0) # Read video

p_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert image from BGR to RGB
    results = pose.process(img_rgb) # Put RGB in for processing
    # print(results.pose_landmarks) # Shows x,y,z co-oridnates (3D Plane) of each landmark and its 'visibility' value
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS) # Draw landmarks on video with connnections between them
        for id, lm in enumerate(results.pose_landmarks.landmark): # Iterate through all landmarks
            h, w, c = img.shape # Height, Width, Channel
            cx, cy = int(lm.x*w), int(lm.y*h) # cx and cy = Pixel values of x and y (2D Plane)
            # Rudolph
            cv2.circle(img, (cx, cy), 17, (0, 0, 255), cv2.FILLED) # Larger visible circle on landmarks
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)

    cv2.waitKey(1)

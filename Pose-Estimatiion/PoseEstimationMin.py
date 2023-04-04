import time

import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# cap = cv2.VideoCapture('PoseVideos/4.mp4')
cap = cv2.VideoCapture(0)

p_time = 0
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    # print(results.pose_landmarks)
    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x*w), int(lm.y*h)
            # Rudolph
            cv2.circle(img, (cx, cy), 17, (0, 0, 255), cv2.FILLED)
    c_time = time.time()
    fps = 1/(c_time - p_time)
    p_time = c_time
    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow("image", img)

    cv2.waitKey(1)

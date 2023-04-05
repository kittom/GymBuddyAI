import cv2
import mediapipe as mp
import time
import math;


class poseDetector():
    def __init__(self, static_image_mode=False,
               model_complexity=1, # Increasing this helped with landmarks going off course but made it significantly slower
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5):

        self.mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = min_detection_confidence
        self.trackCon = min_tracking_confidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth,
                                     self.enable_segmentation, self.smooth_segmentation,
                                     self.detectionCon, self.trackCon)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList
    
    def CalculateAngle(self, img, lm1, lm2, lm3, draw = True):

        # Find Landmarks
        x1, y1 = self.lmList[lm1][1:]; # Get x and y co-ordinates of landmark 1
        x2, y2 = self.lmList[lm2][1:]; # "
        x3, y3 = self.lmList[lm3][1:]; # "

        # Angle Calculation

        angleRad = math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2); 
        angle = math.degrees(angleRad); 

        if angle < 0:
            #angle = angle + 360;
            angle = -angle; # Incase angle is negative, turn it to a positive value


        # Draw lines and landmarks
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3); # Connection between points 1 and 2
            cv2.line(img, (x2, y2), (x3, y3), (0, 255, 0), 3); # Connection between points 2 and 3
            cv2.circle(img, (x1, y1), 6, (255, 0, 255), cv2.FILLED); 
            cv2.circle(img, (x1, y1), 9, (255, 0, 255), 2); # Border around point
            cv2.circle(img, (x2, y2), 6, (255, 0, 255), cv2.FILLED); 
            cv2.circle(img, (x2, y2), 9, (255, 0, 255), 2); 
            cv2.circle(img, (x3, y3), 6, (255, 0, 255), cv2.FILLED); 
            cv2.circle(img, (x3, y3), 9, (255, 0, 255), 2); 
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2); 

        return angle; 


def main():
    cap = cv2.VideoCapture('PoseVideos/ShakilTest.MOV')
    pTime = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img, draw=False)
        if len(lmList) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (0, 0, 255), cv2.FILLED)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()

import cv2;
import numpy;
import time;
import PoseModule;

# Important
Exercise = "Right Bicep Curl"; 
repCount = 0; 
direction = 0; 

video = cv2.VideoCapture("PoseVideos/ShakilTest2.MOV"); 
detector = PoseModule.poseDetector(); 
pTime = 0; 

while True:
    success, img = video.read(); 
    #img = cv2.resize(img, (1280, 720)); 
    img = detector.findPose(img, False); 
    #img = cv2.imread("PoseVideos/ShakilTest.MOV"); 
    lmList = detector.findPosition(img, False); # Get all landmark values

    if len(lmList) != 0:

        if Exercise == "Squat":
            angle = detector.CalculateAngle(img, 12, 14, 16); # Change numbers according to their assigned landmark value
        elif Exercise == "Right Bicep Curl":
            topAngle, bottomAngle = 45, 160; # Specific to this exercise
            angle = detector.CalculateAngle(img, 12, 14, 16); # This one has the correct numbers
            percentage = numpy.interp(angle, (160, 45), (0, 100)); # angle, top/bottom angles, percentage range
        elif Exercise == "Bench Press":
            angle = detector.CalculateAngle(img, 12, 14, 16); 
        else:
            angle = detector.CalculateAngle(img, 12, 14, 16); # ( Right Bicep Curl value )

    # Commented example below from tutorial but doesnt seem to work very well #

    #    if percentage == 100 and direction == 0:
    #        direction = 1; 
    #        repCount = repCount + 0.5;  
    #    if percentage == 0 and direction == 1:
    #        direction = 0; 
    #        repCount = repCount + 0.5;  

        angleThreshold = 0.8; # 80% of the 'optimal' range of motion should be good to be counted as a rep

        if int(angle) <= int(topAngle / angleThreshold) and direction == 0:
            direction = 1; 
            repCount += 0.5; 
        if int(angle) >= int(angleThreshold * bottomAngle) and direction == 1:
            direction = 0; 
            repCount += 0.5; 

        # Display rep count on image
        cv2.putText(img, f"Reps: {repCount}", (20, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2);

    cTime = time.time(); 
    fps = 1/(cTime - pTime); 
    ifps = int(fps); 
    pTime = cTime; 
    cv2.putText(img, f"FPS: {ifps}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 0), 2); 

    cv2.putText(img, f"Range Of Motion: {percentage}", (20, 150), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 200), 2); 

    cv2.imshow("Image", img);
    cv2.waitKey(1); 
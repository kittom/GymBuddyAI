import cv2
import csv
import numpy as np

def draw_keypoints(image, keypoints, radius=10, color=(0, 0, 255)):
    for keypoint in keypoints:
        cv2.circle(image, (keypoint[1], keypoint[2]), radius, color, cv2.FILLED)
    return image

def reconstruct_video_from_csv(csv_path, video_path, fps=30):
    frame_counter = -1
    keypoints_by_frame = {}
    frame_size = None

    # Read the keypoints from the CSV file
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        next(csv_reader)  # Skip the header row

        for row in csv_reader:
            frame, landmark_id, x, y, frame_width, frame_height = int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), int(row[5])

            if frame_size is None:
                frame_size = (frame_width, frame_height)

            if frame != frame_counter:
                keypoints_by_frame[frame] = []
                frame_counter = frame

            keypoints_by_frame[frame].append([landmark_id, x, y])

    # Initialize the video writer
    video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    for frame, keypoints in keypoints_by_frame.items():
        # Create a blank image for each frame
        img = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
        img = draw_keypoints(img, keypoints)
        video_writer.write(img)

    video_writer.release()

if __name__ == "__main__":
    csv_path = "keypoints.csv"
    video_path = "reconstructed_keypoints.mp4"
    reconstruct_video_from_csv(csv_path, video_path)

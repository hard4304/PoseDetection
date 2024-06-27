import cv2
import mediapipe as mp
import numpy as np

mPose = mp.solutions.pose
mpDraw = mp.solutions.drawing_utils
pose = mPose.Pose()

# Use 0 to capture your own movements from the camera itself
# This is how I shall be reading my own video
# cap = cv2.VideoCapture(0)

cap = cv2.VideoCapture('myVideo.mp4')

# This is for the source image dots
drawSpec1 = mpDraw.DrawingSpec(thickness=2, circle_radius=3, color=(255, 0, 0))  # The color is in the BGR format

# This is for the lines
drawSpec2 = mpDraw.DrawingSpec(thickness=3, circle_radius=8, color=(0, 255, 0))

while True:
    # Read a frame from the video or camera
    success, img = cap.read()
    if not success:
        print("Failed to read frame. Exiting...")
        break

    # Resize the image to 800x600
    img = cv2.resize(img, (1000, 1000))

    # Process the image to detect pose
    results = pose.process(img)

    # Draw landmarks on the image (red dots)
    mpDraw.draw_landmarks(img, results.pose_landmarks, mPose.POSE_CONNECTIONS, drawSpec1, drawSpec2)

    # Create a blank image with the same dimensions
    h, w, c = img.shape
    imgBlank = np.zeros([h, w, c])
    imgBlank.fill(255)  # Fill the image with white color

    # Draw landmarks on the blank image
    mpDraw.draw_landmarks(imgBlank, results.pose_landmarks, mPose.POSE_CONNECTIONS, drawSpec1, drawSpec2)

    # Display the images
    cv2.imshow('PoseDetection', img)
    cv2.imshow('The Extracted Pose', imgBlank)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

import cv2
import pickle
import cvzone
import numpy as np

# Load the image
img = cv2.imread('carParkImg.png')  # Use the image file path

# Load the parking slot positions
with open('CarParkPos', 'rb') as f:
    posList = pickle.load(f)

print(posList)
# Define the size of each parking slot (as in the original code)
width, height = 107, 48

# Function to check parking space status
def checkParkingSpace(imgPro):
    spaceCounter = 0

    # Iterate through each parking position
    for pos in posList:
        x, y = pos

        # Crop the image according to the parking slot position
        imgCrop = imgPro[y:y + height, x:x + width]

        # Count non-zero pixels (occupied or empty space)
        count = cv2.countNonZero(imgCrop)

        # If the count is below a threshold, we assume the space is free
        if count < 900:
            color = (0, 255, 0)  # Green for free
            thickness = 5
            spaceCounter += 1
        else:
            color = (0, 0, 255)  # Red for occupied
            thickness = 2

        # Draw a rectangle around the parking slot
        cv2.rectangle(img, pos, (pos[0] + width, pos[1] + height), color, thickness)
        
        # Display the count (free/occupied) on the image
        cvzone.putTextRect(img, str(count), (x, y + height - 3), scale=1, thickness=2, offset=0, colorR=color)

    # Display the total free parking spaces
    cvzone.putTextRect(img, f'Free: {spaceCounter}/{len(posList)}', (100, 50), scale=3, thickness=5, offset=20, colorR=(0, 200, 0))

# Process the image
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (3, 3), 1)
imgThreshold = cv2.adaptiveThreshold(imgBlur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 25, 16)
imgMedian = cv2.medianBlur(imgThreshold, 5)
kernel = np.ones((3, 3), np.uint8)
imgDilate = cv2.dilate(imgMedian, kernel, iterations=1)

# Call the function to check the parking space
checkParkingSpace(imgDilate)

# Show the processed image with the parking space status
cv2.imshow("Parking Space Status", img)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

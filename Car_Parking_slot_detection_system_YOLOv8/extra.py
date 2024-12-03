import cv2

# List to store the coordinates of the rectangles (4 corners each)
rectangles = []

# Variable to track the state of clicks (we need 4 clicks for each rectangle)
click_count = 0
current_rectangle = []

# Function to capture the 4 corner clicks
def draw_rectangle(event, x, y, flags, param):
    global click_count, current_rectangle, rectangles
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if click_count < 4:
            # Add the clicked point to the current rectangle
            current_rectangle.append((x, y))
            click_count += 1
            print(f"Point {click_count} clicked: ({x}, {y})")
        
        # If 4 points are clicked, allow printing the coordinates with spacebar
        if click_count == 4:
            print("Press space to save and print the rectangle coordinates.")
            
# Load the parking lot image
img = cv2.imread("carpark.png")
cv2.imshow("Draw Rectangles", img)
cv2.setMouseCallback("Draw Rectangles", draw_rectangle)

# Wait for the user to draw rectangles on the image
print("Click 4 corners of the rectangle. Press space to print coordinates. Press 'q' to quit.")

while True:
    # Display the image with drawn points and rectangle (if 4 corners are clicked)
    img_copy = img.copy()
    
    if len(current_rectangle) == 4:
        # Draw a rectangle based on the 4 corners
        cv2.polylines(img_copy, [np.array(current_rectangle)], isClosed=True, color=(0, 255, 0), thickness=2)

    for point in current_rectangle:
        cv2.circle(img_copy, point, 5, (0, 0, 255), -1)
    
    cv2.imshow("Draw Rectangles", img_copy)
    
    # Wait for key events
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):  # Press 'q' to quit
        break
    elif key == ord(' '):  # Press space to save and print coordinates
        if click_count == 4:
            # Save the 4 corners as the current rectangle
            rectangles.append(current_rectangle)
            print(f"Rectangle coordinates: {current_rectangle}")
            
            # Reset for the next rectangle
            current_rectangle = []
            click_count = 0
            print("Click 4 corners for the next rectangle.")
        
# Print out the coordinates of all rectangles
print("\nAll Rectangles Coordinates:")
for idx, rect in enumerate(rectangles):
    print(f"Rectangle {idx+1}: {rect}")

# Optionally, save the coordinates to a file
with open("rectangles_coordinates.txt", "w") as f:
    for rect in rectangles:
        f.write(f"Coordinates: {rect}\n")

# Release resources and close windows
cv2.destroyAllWindows()

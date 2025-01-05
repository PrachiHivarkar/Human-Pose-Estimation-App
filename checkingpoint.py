import cv2
import os

## Specify the path to your image
image_path = r"C:\Users\PRACHI\OneDrive\Pictures\ganpati.jpeg"  ## Replace with the actual path to your image

## Debug: Print the current working directory and the image path
print("Current Working Directory:", os.getcwd())
print("Image Path:", image_path)

## Load the image
img = cv2.imread(image_path)

# #Check if the image is loaded successfully
if img is None:
    print("Error: Failed to load image. Check the file path or file format.")
    exit()  ## Exit the program if the image is not loaded

##Display the image
cv2.imshow('color_image', img)
cv2.waitKey(0)  ## Wait for a key press
cv2.destroyAllWindows()  ## Close all OpenCV windows

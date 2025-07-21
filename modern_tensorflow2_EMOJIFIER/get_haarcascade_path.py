import cv2

# Print the directory where Haar cascades are stored
print("Haarcascade directory:", cv2.data.haarcascades)

# Full path to the specific XML file
print("Haarcascade frontal face path:", cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


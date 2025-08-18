import os
import cv2

# Directory to save captured images
DATA_DIR = './data'

# Number of gesture classes (e.g., 26 letters + some extra signs)
number_of_classes = 36

# Number of images to capture per class
dataset_size = 100

# Create main data directory if not exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Open the default camera
cap = cv2.VideoCapture(1)

for class_id in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(class_id))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {class_id}')
    print('Get ready! Press "q" to start capturing images for this class.')

    # Wait for user to press 'q' to start capturing images
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        cv2.putText(frame, 'Ready? Press "q" to start capturing', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Capture dataset_size number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from camera.")
            break

        # Show the frame with a counter
        cv2.putText(frame, f'Capturing class {class_id} image {counter+1}/{dataset_size}', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('frame', frame)

        # Save the captured frame to the class folder
        img_path = os.path.join(class_dir, f'{counter}.jpg')
        cv2.imwrite(img_path, frame)
        counter += 1

        # Delay for smoother capture and allow exit by pressing 'q'
        if cv2.waitKey(100) & 0xFF == ord('q'):
            print('Capture interrupted by user.')
            break

print('Data collection completed.')

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()

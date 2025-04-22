import cv2
import numpy as np
import tensorflow as tf
from tflite_runtime.interpreter import Interpreter

# Load the TFLite model
interpreter = Interpreter(model_path="model_trained.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# List of class labels (update this if your dataset changes)
class_names = [
    'Speed Limit 20 km/h', 'Speed Limit 30 km/h', 'Speed Limit 50 km/h',
    'Speed Limit 60 km/h', 'Speed Limit 70 km/h', 'Speed Limit 80 km/h',
    'End of Speed Limit 80 km/h', 'Speed Limit 100 km/h', 'Speed Limit 120 km/h',
    'No passing', 'No passing for vehicles over 3.5 metric tons',
    'Right-of-way at the next intersection', 'Priority road', 'Yield', 'Stop',
    'No vehicles', 'Vehicles over 3.5 metric tons prohibited', 'No entry',
    'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right',
    'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End of all speed and passing limits', 'Turn right ahead', 'Turn left ahead',
    'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing',
    'End of no passing by vehicles over 3.5 metric tons'
]

# Preprocessing function (grayscale, histogram equalize, normalize, resize)
def preprocess(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.equalizeHist(img)  # Histogram equalization
    img = img / 255.0  # Normalize pixel values
    img = cv2.resize(img, (32, 32))  # Resize to match model input size
    return img.reshape(1, 32, 32, 1).astype(np.float32)

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Width
cap.set(4, 480)  # Height
cap.set(10, 180)  # Brightness

print("📷 Starting Webcam for Real-Time Traffic Sign Detection...")

while True:
    success, frame = cap.read()  # Capture frame-by-frame
    if not success:
        print("❌ Could not read frame from webcam.")
        break

    # Preprocess the current frame
    input_data = preprocess(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Prediction and confidence
    prediction = np.argmax(output_data)  # Index of max output value
    confidence = np.max(output_data)  # Maximum confidence value

    # Display result if confidence is high
    if confidence > 0.75:
        label = f"{prediction}: {class_names[prediction]} ({confidence * 100:.2f}%)"
        cv2.putText(frame, label, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    # Show the output frame
    cv2.imshow("🚦 TinyML Traffic Sign Recognition", frame)

    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

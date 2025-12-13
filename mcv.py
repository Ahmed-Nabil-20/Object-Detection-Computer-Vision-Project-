# Computer Vision 
# Project: Real-time Object Detection using pre-trained YOLOv3 model with OpenCV.
# The system captures video from the webcam and draws bounding boxes around detected objects.
# YOLOv3 was chosen for its speed, accuracy, and ability to detect multiple object types.

import cv2
import numpy as np
import tkinter as tk
from PIL import ImageTk, Image

cfg_file = r'D:\Computer Vision\yolov3.cfg'
weights_file = r'D:\Computer Vision\yolov3.weights'

net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)

classes = []
with open(r'D:\Computer Vision\coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Initialize the GUI
window = tk.Tk()
window.title("Object Detection")
window.geometry("800x600")

# Create a canvas to display video frames
canvas = tk.Canvas(window, width=800, height=600)
canvas.pack()

# Function to perform object detection on a frame
def detect_objects(frame):
    # Preprocess the frame for input to the model
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass through the model
    layer_names = net.getLayerNames()

    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError as e:
        print(f"Error: {e}. Ensure that the model is loaded correctly.")
        return frame

    outputs = net.forward(output_layers)

    # Process the outputs and draw bounding boxes
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * 800)
                center_y = int(detection[1] * 600)
                width = int(detection[2] * 800)
                height = int(detection[3] * 600)
                x = int(center_x - width / 2)
                y = int(center_y - height / 2)

                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, classes[class_id], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

# Function to update the video frame in the GUI
def update_frame():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = detect_objects(frame)

    img = ImageTk.PhotoImage(Image.fromarray(frame))
    canvas.create_image(0, 0, anchor=tk.NW, image=img)
    canvas.image = img
    
    window.after(33, update_frame)  # تحديث بمعدل 30 إطارًا في الثانية تقريبًا

# Open the video capture
cap = cv2.VideoCapture(0)

# Start the video frame update loop
update_frame()

# Start the GUI main loop
window.mainloop()

# Release the video capture
cap.release()

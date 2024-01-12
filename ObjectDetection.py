import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

class ObjectDetector:
    def __init__(self, class_names_file):
        # Load class names from the file
        self.class_names = self.load_class_names(class_names_file)

    def load_class_names(self, class_names_file):
        with open(class_names_file, 'r') as file:
            class_names = [line.strip() for line in file]
        return class_names

    def detect_objects(self, frame, model):
        print('Detection started')
        
        # Convert the image to a tensor
        input_tensor = tf.convert_to_tensor(frame, dtype=tf.uint8)
        input_tensor = tf.expand_dims(input_tensor, 0)  # Add batch dimension

        # Run inference
        detections = model(input_tensor)
        
        # Extract results
        boxes = np.array(detections['detection_boxes'])[0]
        classes = np.array(detections['detection_classes']).astype(int)[0]
        scores = np.array(detections['detection_scores'])[0]

        # Filter out low-confidence detections
        threshold = 0.5
        valid_detections = np.where(scores > threshold)[0]

        # Draw bounding boxes on the image
        for i in valid_detections:
            box = boxes[i]
            class_id = classes[i]
            score = scores[i]
            h, w, _ = frame.shape
            [ymin, xmin, ymax, xmax] = box  # Convert numpy array to list
            xmin = int(xmin * w)
            xmax = int(xmax * w)
            ymin = int(ymin * h)
            ymax = int(ymax * h)

            # Draw bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Get class label using the loaded class names
            class_name = self.class_names[class_id - 1]  # Class IDs are 1-indexed
            label = f"Class: {class_name}, Score: {score:.2f}"
            cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Get a list of detected classes
        detected_classes = [self.class_names[class_id - 1] for class_id in classes[valid_detections]]

        print('Detection ended')
        return frame, detected_classes

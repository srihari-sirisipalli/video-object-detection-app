# Video Object Detection Application

This project is a Python-based video processing and object detection application that utilizes TensorFlow, OpenCV, and Tkinter. The application allows users to select a video, extract frames at a specified minute, perform object detection on the extracted frames, and display the results.

## Project Structure

- **ObjectDetection.py:**
  - Defines the `ObjectDetector` class responsible for loading a pre-trained object detection model and performing object detection on frames.
  - The class includes methods for loading class names and drawing bounding boxes on detected objects.

- **VideoProcessor.py:**
  - Implements the `VideoProcessor` class, which contains methods for extracting frames, displaying frames, and detecting objects in a frame.
  - Utilizes the `ObjectDetector` class for object detection.

- **app.py:**
  - Contains the main Tkinter application, `VideoSelectorApp`.
  - Provides a graphical user interface with buttons for selecting a video, extracting frames, detecting objects, and saving frames.
  - Displays the original and detected frames along with detection results.

## Getting Started

1. Clone the repository to your local machine.

```bash
git clone https://github.com/your-username/video-object-detection-app.git
cd video-object-detection-app
```

2. Install the required libraries.

```bash
pip install -r requirements.txt
```

3. Download the pre-trained model from TensorFlow Hub.

   - Open `__main__` section in `app.py`.
   - Update the model URL in the following line:

     ```python
     model = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/frameworks/TensorFlow2/variations/ssd-mobilenet-v2/versions/1")
     ```

4. Download the class names file.

   - Open `__main__` section in `app.py`.
   - Update the class names file path in the following line:

     ```python
     class_names_path = r'E:/SriHari/Lauretta_Assignment/coco-labels-2014_2017.txt'
     ```

5. Run the application.

```bash
python app.py
```

## Usage

1. Click the "Select Video" button to choose a video file.
2. Choose the desired minute from the dropdown menu.
3. Click the "Extract" button to extract and display the frame.
4. Optionally, click the "Detect" button to perform object detection on the extracted frame.
5. Save the extracted or detected frames using the respective "Save" buttons.

## Dependencies

- TensorFlow
- TensorFlow Hub
- OpenCV
- Pillow (PIL)
- Tkinter

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- The object detection model used in this project is based on the [ssd-mobilenet-v2](https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/frameworks/TensorFlow2/variations/ssd-mobilenet-v2/versions/1) model from TensorFlow Hub.
- Class names are sourced from the [COCO dataset](https://cocodataset.org/#home).


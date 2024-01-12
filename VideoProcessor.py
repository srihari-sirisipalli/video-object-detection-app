import cv2
from PIL import Image, ImageTk
from ObjectDetection import ObjectDetector
class VideoProcessor:
    def __init__(self,model_url):
        self.model_url=model_url
    def extract_frame(path,minute):
        # print(path)
        # print(minute)
        video_capture = cv2.VideoCapture(path)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        target_frame = int(fps * 60 * minute)
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = video_capture.read()
        return frame
    def display_frame(frame):
        frame_copy = frame.copy()
        max_width = 400
        max_height = 300
        height, width, _ = frame_copy.shape
        if width > max_width or height > max_height:
            scale = min(max_width / width, max_height / height)
            frame_copy = cv2.resize(frame, (int(width * scale), int(height * scale)))

        frame_copy_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
        tk_image = ImageTk.PhotoImage(Image.fromarray(frame_copy_rgb))
        return tk_image
    def detect_frame(frame,model,class_path):
        detector=ObjectDetector(class_path)
        detected_frame,objects=detector.detect_objects(frame,model)
        return detected_frame,objects
        

        
        
        
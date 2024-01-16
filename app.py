import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from VideoProcessor import VideoProcessor
import tensorflow_hub as hub


class VideoSelectorApp:
    def __init__(self, master, model,class_names_path):
        self.master = master
        self.model = model
        self.class_names_path=class_names_path
        self.master.title("Video Selector App")

        # Variables to store selected video and minute
        self.selected_video_path = ""
        self.selected_minute = tk.StringVar()

        # Variable to store the original frame
        self.original_frame = None

        # Create a label for the selected video
        self.video_label = tk.Label(self.master, text="Selected Video: None")
        self.video_label.grid(row=0, column=0, columnspan=2, pady=5)

        # Create a button to open the file dialog
        self.select_button = tk.Button(self.master, text="Select Video", command=self.select_video)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=5)

        # Create a label and dropdown menu for selecting minutes
        self.minutes_label = tk.Label(self.master, text="Select Minute:")
        self.minutes_label.grid(row=2, column=0, columnspan=2)

        self.minutes_dropdown = tk.OptionMenu(self.master, self.selected_minute, ())
        self.minutes_dropdown.grid(row=3, column=0, columnspan=2, pady=10)

        # Create a button for extracting and displaying frames
        self.extract_button = tk.Button(self.master, text="Extract", command=self.extract_and_display_frame)
        self.extract_button.grid(row=4, column=0, pady=10)

        # Create a button for detecting faces
        self.detect_button = tk.Button(self.master, text="Detect", command=self.detect_and_display_frame)
        self.detect_button.grid(row=4, column=1, pady=10)

        # Create a label to display the extracted frame
        self.frame_label = tk.Label(self.master)
        self.frame_label.grid(row=5, column=0, pady=10)

        # Create a label to display the detected frame
        self.detected_frame_label = tk.Label(self.master)
        self.detected_frame_label.grid(row=5, column=1, pady=10)

        # Create a label for displaying detection results
        self.results_label = tk.Label(self.master, text="Detection Results:")
        self.results_label.grid(row=6, column=0, columnspan=2, pady=5)

        # Add buttons for saving extracted and detected frames
        self.save_extracted_button = tk.Button(self.master, text="Save Extracted", command=self.save_extracted_frame)
        self.save_extracted_button.grid(row=7, column=0, pady=10)

        self.save_detected_button = tk.Button(self.master, text="Save Detected", command=self.save_detected_frame)
        self.save_detected_button.grid(row=7, column=1, pady=10)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv;*.webm")])

        if file_path:
            video_name = file_path.split("/")[-1]
            self.selected_video_path = file_path
            self.video_label.config(text=f"Selected Video: {video_name}")

            video_capture = cv2.VideoCapture(file_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_duration = int(frame_count / fps // 60) + 1
            video_capture.release()

            self.selected_minute.set("")
            self.minutes_dropdown["menu"].delete(0, "end")
            for minute in range(video_duration):
                self.minutes_dropdown["menu"].add_command(label=str(minute), command=tk._setit(self.selected_minute, minute))
        else:
            messagebox.showinfo("No Video Selected", "You did not select any video.")

    def extract_and_display_frame(self):
        if self.selected_video_path:
            extracted_frame = VideoProcessor.extract_frame(self.selected_video_path, int(self.selected_minute.get()))
            self.tk_image = VideoProcessor.display_frame(extracted_frame)
            self.frame_label.config(image=self.tk_image)
            self.frame_label.image = self.tk_image
            self.extracted_frame = extracted_frame
        else:
            messagebox.showinfo("No Video Selected", "Please select a video before extracting frames.")

    def detect_and_display_frame(self):
        extracted_frame=self.extracted_frame.copy()
        self.detected_frame, objects = VideoProcessor.detect_frame(extracted_frame, self.model,self.class_names_path)
        self.tk_image = VideoProcessor.display_frame(self.detected_frame)
        self.detected_frame_label.config(image=self.tk_image)
        self.detected_frame_label.image = self.tk_image
        self.objects = objects
        self.update_results_label()

    def update_results_label(self):
        objects=self.objects
        obj_dict=dict()
        for ele in objects:
            obj_dict[ele]=objects.count(ele)
        result_text = f"Detection Results: Total Objects: {len(self.objects)}, Unique Objects: {len(set(self.objects))} \n {obj_dict}"
        self.results_label.config(text=result_text)

    def save_frame(self, frame, frame_type):
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    
        if file_path:
            cv2.imwrite(file_path, frame)  # Convert BGR to RGB
            messagebox.showinfo("Save Successful", f"{frame_type.capitalize()} frame saved successfully.")
        else:
            messagebox.showinfo("Save Cancelled", "Save operation cancelled.")


    def save_extracted_frame(self):
        if hasattr(self, 'extracted_frame') and self.extracted_frame is not None:
            self.save_frame(self.extracted_frame, "extracted")
        else:
            messagebox.showinfo("No Extracted Frame", "No frame has been extracted yet.")

    def save_detected_frame(self):
        if hasattr(self, 'detected_frame') and self.detected_frame is not None:
            self.save_frame(self.detected_frame, "detected")
        else:
            messagebox.showinfo("No Detected Frame", "No frame has been detected yet.")

    

if __name__ == "__main__":
    model = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/frameworks/TensorFlow2/variations/ssd-mobilenet-v2/versions/1")
    class_names_path=r'E:/SriHari/Lauretta_Assignment/coco-labels-2014_2017.txt'
    root = tk.Tk()
    app = VideoSelectorApp(root, model,class_names_path)
    root.mainloop()

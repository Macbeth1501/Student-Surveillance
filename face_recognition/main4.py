import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageTk
import numpy as np
from facenet_pytorch import MTCNN
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import threading

# Define the model architecture
class FaceRecognitionResNet(nn.Module):
    def __init__(self, num_classes, freeze_layers=True):
        super(FaceRecognitionResNet, self).__init__()
        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        
        if freeze_layers:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.resnet(x)

def load_label_map(dataset_path):
    """Load the label map from labels.csv."""
    labels_df = pd.read_csv(os.path.join(dataset_path, "labels.csv"))
    label_map = dict(zip(labels_df["Id"], labels_df["Name"]))
    return label_map, len(label_map)

def predict_faces_in_frame(frame, model, mtcnn, label_map, device):
    """Predict identities for faces in a frame (image or video)."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    boxes, _ = mtcnn.detect(pil_img)
    predictions = []
    
    if boxes is not None:
        for box in boxes:
            face = mtcnn.extract(pil_img, [box], None)
            if face is not None and len(face) > 0:
                face = face[0].to(device)
                face = face.unsqueeze(0)
                
                with torch.no_grad():
                    output = model(face)
                    _, predicted = torch.max(output.data, 1)
                    predicted_id = predicted.item()
                    predicted_name = label_map[predicted_id]
                    predictions.append((predicted_name, box))
    
    return predictions

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("800x600")

        # Paths and configurations
        self.dataset_path = "C:/Users/Rochan/Desktop/Coding/Student_Surveillance/face_recognition/Data"
        self.model_path = "best_model.pth"
        
        # Create outputs folder if it doesn't exist
        self.output_dir = "outputs"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.output_photo_path = os.path.join(self.output_dir, "output_photo.jpg")
        self.output_video_path = os.path.join(self.output_dir, "output.mp4")
        
        # Load label map
        self.label_map, self.num_classes = load_label_map(self.dataset_path)
        print(f"Label map: {self.label_map}")
        print(f"Number of classes: {self.num_classes}")

        # Initialize device and MTCNN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size=160, margin=0, min_face_size=15, thresholds=[0.4, 0.5, 0.5], 
                           keep_all=True, device=self.device)
        print(f"Using device: {self.device}")

        # Initialize and load the model
        self.model = FaceRecognitionResNet(num_classes=self.num_classes, freeze_layers=True).to(self.device)
        if not os.path.exists(self.model_path):
            messagebox.showerror("Error", f"Model file {self.model_path} not found.")
            self.root.destroy()
            return
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()
        print("Model loaded successfully.")

        # GUI elements
        # Status label at the top
        self.status_label = tk.Label(self.root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(pady=10)

        # Frame for upload buttons (first row)
        self.upload_frame = tk.Frame(self.root)
        self.upload_frame.pack(pady=5)

        # Photo upload button
        self.upload_photo_button = tk.Button(self.upload_frame, text="Upload Photo", command=self.upload_photo, font=("Arial", 12))
        self.upload_photo_button.grid(row=0, column=0, padx=5)

        # Video upload button
        self.upload_video_button = tk.Button(self.upload_frame, text="Upload Video", command=self.upload_video, font=("Arial", 12))
        self.upload_video_button.grid(row=0, column=1, padx=5)

        # Frame for detection buttons (second row)
        self.detection_frame = tk.Frame(self.root)
        self.detection_frame.pack(pady=5)

        # Process photo button
        self.process_photo_button = tk.Button(self.detection_frame, text="Process Photo", command=self.process_photo, font=("Arial", 12))
        self.process_photo_button.grid(row=0, column=0, padx=5)
        self.process_photo_button.config(state=tk.DISABLED)

        # Start video detection button
        self.start_video_button = tk.Button(self.detection_frame, text="Start Video Detection", command=self.start_video_detection, font=("Arial", 12))
        self.start_video_button.grid(row=0, column=1, padx=5)
        self.start_video_button.config(state=tk.DISABLED)

        # Start webcam button
        self.start_webcam_button = tk.Button(self.detection_frame, text="Start Webcam", command=self.start_webcam, font=("Arial", 12))
        self.start_webcam_button.grid(row=0, column=2, padx=5)

        # Start IP webcam button
        self.start_ip_webcam_button = tk.Button(self.detection_frame, text="Start IP Webcam", command=self.prompt_ip_webcam, font=("Arial", 12))
        self.start_ip_webcam_button.grid(row=0, column=3, padx=5)

        # Stop detection button
        self.stop_button = tk.Button(self.detection_frame, text="Stop Detection", command=self.stop_detection, font=("Arial", 12))
        self.stop_button.grid(row=0, column=4, padx=5)
        self.stop_button.config(state=tk.DISABLED)

        # Canvas for displaying photo, video, webcam, or IP webcam feed
        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack(pady=10)

        # Variables
        self.cap = None
        self.out = None
        self.is_running = False
        self.thread = None
        self.is_webcam = False
        self.is_ip_webcam = False

    def upload_photo(self):
        """Upload a photo file."""
        file_path = filedialog.askopenfilename(
            title="Select Photo",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]
        )
        if file_path:
            self.photo_path = file_path
            self.status_label.config(text=f"Status: Photo selected - {os.path.basename(file_path)}")
            self.process_photo_button.config(state=tk.NORMAL)

    def process_photo(self):
        """Process the uploaded photo and display the result."""
        if not hasattr(self, 'photo_path'):
            messagebox.showerror("Error", "Please upload a photo first.")
            return

        # Load and process the photo
        frame = cv2.imread(self.photo_path)
        if frame is None:
            messagebox.showerror("Error", "Could not load photo.")
            self.status_label.config(text="Status: Error - Could not load photo")
            return

        # Predict faces in the photo
        predictions = predict_faces_in_frame(frame, self.model, self.mtcnn, self.label_map, self.device)

        # Draw bounding boxes and labels
        for predicted_name, box in predictions:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_name, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the processed photo in the outputs folder
        cv2.imwrite(self.output_photo_path, frame)

        # Display the processed photo on the canvas
        self.display_frame(frame)
        self.status_label.config(text=f"Status: Photo Processed. Saved to {self.output_photo_path}")

    def upload_video(self):
        """Upload a video file."""
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )
        if file_path:
            self.video_source = file_path
            self.status_label.config(text=f"Status: Video selected - {os.path.basename(file_path)}")
            self.start_video_button.config(state=tk.NORMAL)

    def start_video_detection(self):
        """Start video face detection in a separate thread."""
        if not hasattr(self, 'video_source'):
            messagebox.showerror("Error", "Please upload a video file first.")
            return

        self.is_webcam = False
        self.is_ip_webcam = False
        self.is_running = True
        self.start_video_button.config(state=tk.DISABLED)
        self.start_webcam_button.config(state=tk.DISABLED)
        self.start_ip_webcam_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Video Processing...")

        # Start video processing in a separate thread
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

    def start_webcam(self):
        """Start default webcam feed for real-time face recognition."""
        self.is_webcam = True
        self.is_ip_webcam = False
        self.video_source = 0  # Default webcam
        self.is_running = True
        self.start_video_button.config(state=tk.DISABLED)
        self.start_webcam_button.config(state=tk.DISABLED)
        self.start_ip_webcam_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Webcam Running...")

        # Start webcam processing in a separate thread
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

    def prompt_ip_webcam(self):
        """Prompt the user to enter the IP webcam address and start the feed."""
        ip_address = simpledialog.askstring("IP Webcam Address", "Enter the IP webcam address (e.g., 192.168.29.166:8080):",
                                            parent=self.root)
        if ip_address:
            # Construct the stream URL
            if not ip_address.startswith("http://"):
                ip_address = "http://" + ip_address
            if not ip_address.endswith("/video"):
                ip_address = ip_address + "/video"
            self.video_source = ip_address
            self.start_ip_webcam()
        else:
            self.status_label.config(text="Status: IP Webcam connection cancelled")

    def start_ip_webcam(self):
        """Start IP webcam feed for real-time face recognition."""
        self.is_webcam = False
        self.is_ip_webcam = True
        self.is_running = True
        self.start_video_button.config(state=tk.DISABLED)
        self.start_webcam_button.config(state=tk.DISABLED)
        self.start_ip_webcam_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_label.config(text="Status: IP Webcam Running...")

        # Start IP webcam processing in a separate thread
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

    def stop_detection(self):
        """Stop the video, webcam, or IP webcam processing."""
        self.is_running = False
        self.start_video_button.config(state=tk.NORMAL)
        self.start_webcam_button.config(state=tk.NORMAL)
        self.start_ip_webcam_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")

    def process_video(self):
        """Process the video, webcam, or IP webcam feed and perform face recognition."""
        self.cap = cv2.VideoCapture(self.video_source)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open video source. Ensure the IP webcam server is running and accessible.")
            self.is_running = False
            self.start_video_button.config(state=tk.NORMAL)
            self.start_webcam_button.config(state=tk.NORMAL)
            self.start_ip_webcam_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.status_label.config(text="Status: Error - Could not open video source")
            return

        # Initialize video writer in the outputs folder
        height, width, _ = self.cap.read()[1].shape
        self.out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Predict faces in the frame
            predictions = predict_faces_in_frame(frame, self.model, self.mtcnn, self.label_map, self.device)

            # Draw bounding boxes and labels
            for predicted_name, box in predictions:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, predicted_name, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Write frame to output video
            self.out.write(frame)

            # Update canvas with the current frame
            self.display_frame(frame)

        # Release resources
        self.cap.release()
        self.out.release()
        self.is_running = False
        self.start_video_button.config(state=tk.NORMAL)
        self.start_webcam_button.config(state=tk.NORMAL)
        self.start_ip_webcam_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        source_type = "IP Webcam" if self.is_ip_webcam else "Webcam" if self.is_webcam else "Video"
        self.status_label.config(text=f"Status: {source_type} Stopped. Output saved to {self.output_video_path}")

    def display_frame(self, frame):
        """Display the frame (photo, video, webcam, or IP webcam) on the Tkinter canvas."""
        # Resize frame to fit canvas
        frame = cv2.resize(frame, (640, 480))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.canvas.imgtk = imgtk
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)

        # Update GUI
        self.root.update()

    def run(self):
        """Run the Tkinter main loop."""
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    app.run()
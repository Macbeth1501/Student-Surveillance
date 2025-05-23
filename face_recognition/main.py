import os
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from facenet_pytorch import MTCNN
import cv2
import argparse

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
    """Predict identities for faces in a video frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    
    # Detect faces and get bounding boxes
    boxes, _ = mtcnn.detect(pil_img)
    predictions = []
    
    if boxes is not None:
        for box in boxes:
            # Extract individual face using MTCNN
            face = mtcnn.extract(pil_img, [box], None)
            if face is not None and len(face) > 0:
                face = face[0].to(device)  # Take the first (and only) face
                face = face.unsqueeze(0)   # Add batch dimension: [1, 3, 160, 160]
                
                # Predict identity for this face
                with torch.no_grad():
                    output = model(face)
                    _, predicted = torch.max(output.data, 1)
                    predicted_id = predicted.item()
                    predicted_name = label_map[predicted_id]
                    predictions.append((predicted_name, box))
    
    return predictions

def main(video_source, use_gui=True):
    """Main function for real-time face recognition."""
    # Paths and configurations
    dataset_path = "C:/Users/Rochan/Desktop/Coding/Student_Surveillance/face_recognition/Data"
    model_path = "best_model.pth"
    output_video_path = "output.mp4"

    # Load label map
    label_map, num_classes = load_label_map(dataset_path)
    print(f"Label map: {label_map}")
    print(f"Number of classes: {num_classes}")

    # Initialize device and MTCNN
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=15, thresholds=[0.4, 0.5, 0.5], 
                  keep_all=True, device=device)
    print(f"Using device: {device}")

    # Initialize and load the model
    model = FaceRecognitionResNet(num_classes=num_classes, freeze_layers=True).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file {model_path} not found.")
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded successfully.")

    # Initialize video capture
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise RuntimeError("Error: Could not open video capture.")
    
    # Initialize video writer
    height, width, _ = cap.read()[1].shape
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))
    
    print("Starting video capture. Press 'q' to quit if using GUI.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Predict faces in the frame
        predictions = predict_faces_in_frame(frame, model, mtcnn, label_map, device)
        
        # Draw bounding boxes and labels
        for predicted_name, box in predictions:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_name, (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        
        # Display the frame if using GUI
        if use_gui:
            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release resources
    cap.release()
    out.release()
    if use_gui:
        cv2.destroyAllWindows()
    print("Video capture stopped. Output saved to", output_video_path)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Real-time face recognition system.")
    parser.add_argument('--video-source', type=str, default='0',
                        help="Video source: '0' for webcam, or path to video file.")
    parser.add_argument('--no-gui', action='store_true',
                        help="Disable GUI display and only save output to video file.")
    args = parser.parse_args()

    # Handle video source argument
    try:
        video_source = int(args.video_source)
    except ValueError:
        video_source = args.video_source

    # Run the main function
    try:
        main(video_source, use_gui=not args.no_gui)
    except Exception as e:
        print(f"An error occurred: {e}")
import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import pyttsx3
import tkinter as tk
from tkinter import Label, Button

# Settings
img_size = 128
confidence_threshold = 0.95
required_stable_frames = 15
class_names = ['background', 'bulbasaur', 'charmander', 'pikachu', 'squirtle']

# Text-to-Speech
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Model
model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(64 * (img_size // 4) * (img_size // 4), 128), nn.ReLU(),
    nn.Linear(128, len(class_names))
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.load_state_dict(torch.load('simple_pokedex_model.pth', map_location=device))
model.to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_from_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    outputs = model(tensor)
    probs = nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()[0]
    idx = np.argmax(probs)
    return class_names[idx], probs[idx]

# UI Setup
root = tk.Tk()
root.title("Pokédex Scanner")
root.geometry("400x200")

result_label = Label(root, text="Click 'Scan Pokémon' to begin.", font=("Arial", 14))
result_label.pack(pady=20)

def scan_pokemon():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        result_label.config(text="Error: Cannot access camera.")
        return

    frame_count = 0
    prev_name = None
    last_spoken = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        name, confidence = predict_from_frame(frame)

        if confidence > confidence_threshold and name != "background":
            if name == prev_name:
                frame_count += 1
            else:
                prev_name = name
                frame_count = 1
        else:
            prev_name = None
            frame_count = 0

        if frame_count >= required_stable_frames:
            label = f"{prev_name.capitalize()} ({confidence * 100:.1f}%)"
            result_label.config(text=f"Detected: {label}")
            if last_spoken != prev_name:
                engine.say(prev_name)
                engine.runAndWait()
                last_spoken = prev_name
            break

        cv2.imshow("Scanning...", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

scan_button = Button(root, text="Scan Pokémon", command=scan_pokemon, font=("Arial", 12))
scan_button.pack(pady=10)

root.mainloop()

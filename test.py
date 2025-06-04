import tkinter as tk
from PIL import Image, ImageTk
import cv2

# Main App Class
class PokedexScannerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pokédex Scanner")

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        if not self.cap.isOpened():
            print("❌ Failed to open camera.")
            return

        self.update_frame()  # Start looping

    def update_frame(self):
        ret, frame = self.cap.read()
        print("Frame captured:", ret, "Frame shape:" if ret else "No frame")

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)

            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        self.root.after(15, self.update_frame)

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()

# Run the App
if __name__ == "__main__":
    root = tk.Tk()
    app = PokedexScannerApp(root)
    root.mainloop()

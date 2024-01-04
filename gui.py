import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.simpledialog import askstring
import cv2
import os
from PIL import Image, ImageTk
from MTCNN import *

class FaceCaptureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Capture App")
        self.root.geometry("300x100")

        self.label = tk.Label(root)
        self.label.grid(row=0, columnspan=3, padx=10, pady=10)

        
        self.camera = None  # Initialize camera variable
        self.camera_opened = False  # Camera status flag
        
        style = ttk.Style()
        style.configure("Custom.TButton", foreground="blue", font=("Helvetica", 12), width=10)

        self.camera_button = tk.Button(root, text="Open Camera", command=self.toggle_camera, fg = "blue")
        self.camera_button.grid(row=2, column=0, padx=10, pady=10, sticky="s")
        
        self.mtcnn_button = tk.Button(root, text="MTCNN", command=self.mtcnn, fg = "blue")
        self.mtcnn_button.grid(row=2, column=2, padx=10, pady=10, sticky="s")
        
    def close_camera(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
            self.camera_opened = False
            self.camera_button.config(text="Open Camera")
            self.crop.grid_forget()

    def mtcnn(self):
        if not self.camera_opened:
            camera_detect()
            # self.mtcnn_button.config(state='disabled')

    def open_camera(self):
        if (self.camera is None):
            self.camera = cv2.VideoCapture(0)
            self.camera_opened = True
            self.update_frame()
            self.root.geometry("750x600")
            # self.camera_button = tk.Button(root, text="Close Face", command=self.close_camera, fg = "blue")
            # self.camera_button.grid(row=2, column=0, padx=10, pady=10, sticky="s") 
            self.camera_button.config(text='Close Camera')
            self.crop = tk.Button(root, text="Crop Camera", command=self.crop_face, fg = "blue")
            self.crop.grid(row=2, column=1, padx=10, pady=10, sticky="s")

    def update_frame(self):
        if self.camera is not None:
            ret, frame = self.camera.read()
            if ret:
                opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                captured_image = Image.fromarray(opencv_image)
                photo_image = ImageTk.PhotoImage(image=captured_image)
                self.label.photo_image = photo_image
                self.label.configure(image=photo_image)
            self.label.after(5, self.update_frame)

    def crop_face(self):
        name = askstring("Capture Face", "Enter the name:")
        if not name:
            messagebox.showerror("Error", "Please enter a name.")
            return
    
        size = askstring("Capture Face", "Size of your face:")
        while size is None:
            messagebox.showerror("Error", "Please enter a name.")
            size = askstring("Capture Face", "Size of your face:")

        directory = f"faces/{name}"
        if not os.path.exists(directory):
            os.makedirs(directory)

        ret, frame = self.camera.read()
        if ret:
            face_image_path = os.path.join(directory, f"{name}.jpg")
            cv2.imwrite(face_image_path, frame)
            detectFace(directory, 'crop/', int(size))
        else:
            messagebox.showerror("Error", "Failed to capture face.")
        
    def toggle_camera(self):
        if self.camera_opened:
            self.close_camera()
        else:
            self.open_camera()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceCaptureApp(root)
    root.mainloop()
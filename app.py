import tkinter as tk
from tkinter import filedialog, Label
import numpy as np
from PIL import Image, ImageTk
from predictTumor import *

class TumorDetection:
    def __init__(self):
        # Initialize a dictionary to keep track of tumor counts
        self.tumor_counts = {
            'glioma': 0,
            'meningioma': 0,
            'pituitary': 0,
            'notumor': 0
        }

    def evaluate_type_for_image(self, image_path, max_size=(400, 400)):
        # Open the image using PIL
        image = Image.open(image_path)
        mri_image = cv.imread(str(image_path), 1)
        # Tumor is run through the model
        res = predictTumor(mri_image)

        max_prob = np.max(res)
        detected_tumor_type = None

        if max_prob > 0.8:
            detected_tumor_type = ['glioma', 'meningioma', 'pituitary', 'notumor'][res.argmax()]
            self.tumor_counts[detected_tumor_type] += 1

        # Resize the image with LANCZOS filter
        image.thumbnail(max_size, Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        return detected_tumor_type, photo

def browse_file():
    # Let user choose a file
    file_path = filedialog.askopenfilename()
    if file_path:
        detected_tumor_type, photo = detector.evaluate_type_for_image(file_path)
        if detected_tumor_type:
            update_display(file_path, detected_tumor_type)

def update_display(image_path, tumor_type):
    original_image = Image.open(image_path)
    max_size = (300, 300)
    resized_image = original_image.resize(max_size, Image.LANCZOS)

    # Create a Label with a grey background
    image_label = Label(root, bg="grey")

    photo = ImageTk.PhotoImage(resized_image)
    image_label.configure(image=photo)
    image_label.image = photo

    # Remove the previous image
    if hasattr(display_label, 'image'):
        display_label.image = None

    # Use grid to position the image label
    image_label.grid(row=0, column=0, padx=20, pady=20, columnspan=2,
                     sticky="nsew")

    if tumor_type == "notumor":
        tumor_type_label.config(text=f"Tumor Type: No Tumor Detected.")
    elif tumor_type == "glioma":
        tumor_type_label.config(text=f"Tumor Type: Glioma Tumor Found.")
    elif tumor_type == "pituitary":
        tumor_type_label.config(text=f"Tumor Type: Pituitary Tumor Found.")
    elif tumor_type == "meningioma":
        tumor_type_label.config(text=f"Tumor Type: Meningioma Tumor Found.")

    tumor_type_label.grid(row=1, column=0, padx=20, pady=20, columnspan=2,
                          sticky="nsew")

def show_about():
    about_text = ("""This app accepts image input in the form of a brain MRI and can detect the presence of 
                     three different kinds of tumors: Glioblastoma, Meningioma, and Pituitary, as well as detecting images 
                     with no tumor. It was developed using Python, Tensorflow, Xception for machine learning, 
                     and TKinter for the visual app. Created by Marshall Morgan for HackRice13. This is not medical advice,
                     and it should not replace consultation with a doctor.""")
    about_text_label.config(text=about_text)

if __name__ == "__main__":
    detector = TumorDetection()

    root = tk.Tk()
    root.title("Tumor Detection")

    # Set the window size
    root.geometry("1280x800")

    # Set the background color
    root.configure(bg='grey')

    # Use grid to center the elements vertically and horizontally
    root.grid_rowconfigure(0, weight=1)
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(0, weight=1)

    display_label = Label(root)
    tumor_type_label = Label(root, text="", font=("Helvetica", 28), fg="black", bg="grey", padx=10, pady=10, borderwidth=2)

    # Place the Browse Button
    browse_button = tk.Button(root, text="Choose An Image", font=("Helvetica", 14), command=browse_file, width=10, height=6, bg="light gray")

    browse_button.grid(row=2, column=0, padx=500, pady=10, columnspan=2,
                       sticky="nsew")

    # Add a Label for the About section
    about_text_label = Label(root, text="", font=("Helvetica", 14), padx=80, pady=10, bg="grey")
    about_text_label.grid(row=3, column=0, padx=20, pady=30, columnspan=2,
                          sticky="nsew")

    # Show the About section
    show_about()

    root.mainloop()

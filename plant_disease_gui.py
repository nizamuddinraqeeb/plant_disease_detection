import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.vgg19 import preprocess_input # type: ignore
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# --- Class Labels (Make sure this list is correct and in the right order) ---
CLASS_LABELS = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___healthy',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___healthy',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___healthy',
    'Potato___Late_blight',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___healthy',
    'Strawberry___Leaf_scorch',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___healthy',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus'
]

# --- Main Application Class ---
class PlantClassifierApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Plant Disease Classifier")
        self.geometry("800x650")
        self.configure(bg="#f8f9fa")

        self.model = None
        self.image_path = ""
        self.MODEL_FILE = 'best_model.h5'

        self.create_widgets()
        self.load_model()
    
    def load_model(self):
        """Loads the pre-trained Keras model."""
        try:
            self.model = tf.keras.models.load_model(self.MODEL_FILE)
            self.status_label.config(text="Model loaded successfully!", fg="#28a745")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model: {e}")
            self.status_label.config(text=f"Failed to load model: {e}", fg="#dc3545")
            self.model = None

    def create_widgets(self):
        """Creates all GUI widgets and sets up the layout."""
        # Main frame to hold the canvas and scrollbar
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a Canvas widget with a scrollbar
        canvas = tk.Canvas(main_frame, bg="#f8f9fa")
        scrollbar = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#f8f9fa")

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # --- Header Frame ---
        header_frame = tk.Frame(scrollable_frame, bg="#34495e", padx=20, pady=15)
        header_frame.pack(fill=tk.X)
        title_label = tk.Label(header_frame, text="Plant Disease Classifier", font=("Helvetica", 28, "bold"), fg="white", bg="#34495e")
        title_label.pack()

        # --- Main Content Frame ---
        content_frame = tk.Frame(scrollable_frame, bg="#f8f9fa", padx=30, pady=30)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # --- Image Display Frame ---
        image_frame = tk.Frame(content_frame, bg="#e9ecef", bd=2, relief="groove")
        image_frame.pack(pady=(20, 0), fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(image_frame, bg="#e9ecef")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_label.bind("<Configure>", self.on_image_frame_resize)

        self.path_label = tk.Label(content_frame, text="Select an image to get started.", font=("Helvetica", 12), bg="#f8f9fa", fg="#6c757d")
        self.path_label.pack(pady=(20, 20))

        # --- Buttons Frame ---
        button_frame = tk.Frame(content_frame, bg="#f8f9fa")
        button_frame.pack(pady=(0, 20))

        style = {
            "font": ("Helvetica", 12, "bold"),
            "fg": "white",
            "bd": 0,
            "relief": "flat",
            "width": 15,
            "height": 2,
            "cursor": "hand2"
        }

        self.create_styled_button(button_frame, "Browse Image", self.browse_image, "#17a2b8", **style)
        self.create_styled_button(button_frame, "Classify", self.classify_image, "#28a745", **style)
        self.create_styled_button(button_frame, "Next Image", self.reset_app, "#ffc107", **style)
        self.create_styled_button(button_frame, "Exit", self.destroy, "#dc3545", **style)

        # --- Status & Result Labels ---
        self.status_label = tk.Label(content_frame, text="", font=("Helvetica", 10, "italic"),
                                     bg="#f8f9fa", fg="#6c757d")
        self.status_label.pack(pady=(0, 10))

        self.result_label = tk.Label(
            content_frame,
            text="",
            font=("Helvetica", 16, "bold"),
            justify=tk.CENTER,
            wraplength=750,
            anchor="center",
            bg="#f8f9fa"
        )
        self.result_label.pack(pady=(10, 0), fill=tk.X, ipadx=10, ipady=20)
        self.result_label.config(height=4)

    def create_styled_button(self, parent, text, command, bg_color, **kwargs):
        """Helper function to create a consistently styled button with hover effects."""
        button = tk.Button(parent, text=text, command=command, bg=bg_color, **kwargs)
        button.pack(side=tk.LEFT, padx=10)
        button.bind("<Enter>", lambda e: button.config(bg="#5a6268"))
        button.bind("<Leave>", lambda e: button.config(bg=bg_color))
        return button

    def on_image_frame_resize(self, event):
        """Resizes the image to fit the frame while maintaining aspect ratio."""
        if self.image_path:
            self.display_image(self.image_path)

    def browse_image(self):
        """Open a file dialog to select an image."""
        file_path = filedialog.askopenfilename(
            title="Select an image file",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
        )
        if file_path:
            self.image_path = file_path
            self.path_label.config(text=f"Selected File: {os.path.basename(self.image_path)}")
            self.display_image(self.image_path)
            self.result_label.config(text="")
            self.status_label.config(text="Ready to classify.", fg="#6c757d")

    def display_image(self, path):
        """Displays the selected image in the GUI, resized to fit."""
        try:
            img = Image.open(path)
            frame_width = self.image_label.winfo_width()
            frame_height = self.image_label.winfo_height()
            
            if frame_width == 1 and frame_height == 1:
                # If the frame hasn't been rendered, use a default size
                frame_width, frame_height = 400, 300

            img.thumbnail((frame_width, frame_height), Image.Resampling.LANCZOS)
            
            img_tk = ImageTk.PhotoImage(img)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk
        except Exception as e:
            messagebox.showerror("Image Error", f"Could not display image: {e}")
            self.status_label.config(text=f"Image Error: {e}", fg="#dc3545")

    def classify_image(self):
        """Classifies the selected image and displays the result."""
        if not self.image_path:
            self.status_label.config(text="Please select an image first.", fg="#ffc107")
            return

        if self.model is None:
            self.status_label.config(text="Model not loaded. Please restart the application.", fg="#dc3545")
            return

        try:
            self.status_label.config(text="Classifying image...", fg="#007bff")
            self.update_idletasks()
            
            img = load_img(self.image_path, target_size=(256, 256))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            processed_image = preprocess_input(img_array)

            predictions = self.model.predict(processed_image, verbose=0)
            predicted_class_index = np.argmax(predictions)
            predicted_class_name = CLASS_LABELS[predicted_class_index]
            prediction_probability = predictions[0][predicted_class_index] * 100

            self.result_label.config(text=f"Prediction: {predicted_class_name}\nConfidence: {prediction_probability:.2f}%", fg="#007bff")
            self.status_label.config(text="Classification complete.", fg="#28a745")
        except Exception as e:
            messagebox.showerror("Classification Error", f"An error occurred during classification: {e}")
            self.status_label.config(text=f"Classification Error: {e}", fg="#dc3545")

    def reset_app(self):
        """Resets the GUI to allow for a new classification."""
        self.image_path = ""
        self.path_label.config(text="Select an image to get started.")
        self.image_label.config(image=None)
        self.result_label.config(text="")
        self.status_label.config(text="Ready to classify another image.", fg="#6c757d")

# --- Main script execution ---
if __name__ == "__main__":
    app = PlantClassifierApp()
    app.mainloop()
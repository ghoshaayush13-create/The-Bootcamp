import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import cv2
import os


# Load YOLO model
model = YOLO("yolo26n.pt")

selected_image = None


# ==========================================
# MAIN WINDOW
# ==========================================

root = tk.Tk()
root.title("AI Object Detection System")
root.geometry("1100x750")
root.configure(bg="#202020")


# ==========================================
# TITLE
# ==========================================

title = tk.Label(
    root,
    text="AI OBJECT DETECTION SYSTEM",
    font=("Arial", 24, "bold"),
    fg="white",
    bg="#202020"
)

title.pack(pady=15)


subtitle = tk.Label(
    root,
    text="Object Detection using YOLO",
    font=("Arial", 12),
    fg="lightgray",
    bg="#202020"
)

subtitle.pack()


# ==========================================
# IMAGE DISPLAY
# ==========================================

image_frame = tk.Frame(
    root,
    bg="#303030",
    width=800,
    height=430
)

image_frame.pack(pady=15)

image_frame.pack_propagate(False)


image_label = tk.Label(
    image_frame,
    text="No image selected",
    font=("Arial", 18),
    fg="white",
    bg="#303030"
)

image_label.pack(
    expand=True
)


# ==========================================
# BUTTON FRAME
# ==========================================

button_frame = tk.Frame(
    root,
    bg="#202020"
)

button_frame.pack(pady=10)


# ==========================================
# SELECT IMAGE
# ==========================================

def select_image():

    global selected_image

    file_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[
            ("Image Files", "*.jpg *.jpeg *.png *.webp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    selected_image = file_path

    image = Image.open(file_path)

    image.thumbnail((780, 420))

    photo = ImageTk.PhotoImage(image)

    image_label.config(
        image=photo,
        text=""
    )

    image_label.image = photo

    result_label.config(
        text="Image selected. Click Detect Objects."
    )


# ==========================================
# DETECT OBJECTS
# ==========================================

def detect_objects():

    if selected_image is None:

        messagebox.showwarning(
            "No Image",
            "Please select an image first."
        )

        return

    result_label.config(
        text="Detecting objects..."
    )

    root.update()

    # Run YOLO
    results = model.predict(
        source=selected_image,
        conf=0.25
    )

    result = results[0]

    # Draw bounding boxes
    output = result.plot()

    # Convert BGR to RGB
    output = cv2.cvtColor(
        output,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL image
    image = Image.fromarray(output)

    image.thumbnail((780, 420))

    photo = ImageTk.PhotoImage(image)

    image_label.config(
        image=photo,
        text=""
    )

    image_label.image = photo


    # Detection information

    detected_objects = []

    for box in result.boxes:

        class_id = int(box.cls[0])

        confidence = float(box.conf[0])

        object_name = model.names[class_id]

        detected_objects.append(
            f"{object_name} ({confidence:.2f})"
        )


    # Display results

    if detected_objects:

        result_text = (
            "Detected Objects: "
            + ", ".join(detected_objects)
        )

    else:

        result_text = "No objects detected."


    result_label.config(
        text=result_text
    )


# ==========================================
# BUTTONS
# ==========================================

select_button = tk.Button(
    button_frame,
    text="SELECT IMAGE",
    command=select_image,
    font=("Arial", 12, "bold"),
    width=20,
    height=2
)

select_button.grid(
    row=0,
    column=0,
    padx=20
)


detect_button = tk.Button(
    button_frame,
    text="DETECT OBJECTS",
    command=detect_objects,
    font=("Arial", 12, "bold"),
    width=20,
    height=2
)

detect_button.grid(
    row=0,
    column=1,
    padx=20
)


# ==========================================
# RESULT LABEL
# ==========================================

result_label = tk.Label(
    root,
    text="Select an image to begin",
    font=("Arial", 12),
    fg="white",
    bg="#202020",
    wraplength=1000
)

result_label.pack(
    pady=10
)


# ==========================================
# START APPLICATION
# ==========================================

root.mainloop()
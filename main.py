# import cv2
# import numpy as np
# import os

# def blur_faces_in_folder(input_folder, output_folder, blur_intensity, confidence_threshold):
#     # Load the DNN face detector model
#     net = cv2.dnn.readNetFromCaffe(
#         'deploy.prototxt', 
#         'res10_300x300_ssd_iter_140000_fp16.caffemodel'
#     )
    
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Loop through all files in the input folder
#     for filename in os.listdir(input_folder):
#         file_path = os.path.join(input_folder, filename)

#         # Check if the file is an image
#         if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
#             image = cv2.imread(file_path)
#             if image is None:
#                 print(f"Error reading {file_path}, skipping.")
#                 continue

#             h, w = image.shape[:2]
#             blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#             net.setInput(blob)
#             detections = net.forward()

#             for i in range(detections.shape[2]):
#                 confidence = detections[0, 0, i, 2]
#                 if confidence > confidence_threshold:
#                     box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                     (x, y, x1, y1) = box.astype("int")

#                     # Ensure the bounding box is within image bounds
#                     x, y = max(0, x), max(0, y)
#                     x1, y1 = min(w, x1), min(h, y1)

#                     if x < x1 and y < y1 and image[y:y1, x:x1].size > 0:
#                         face = image[y:y1, x:x1]
#                         blurred_face = cv2.GaussianBlur(face, (blur_intensity, blur_intensity), 30)
#                         image[y:y1, x:x1] = blurred_face

#             output_path = os.path.join(output_folder, filename)
#             cv2.imwrite(output_path, image)
#             print(f"Processed and saved: {output_path}")

# if __name__ == "__main__":
#     input_folder = input("Enter the input folder path: ").strip()
#     output_folder = input("Enter the output folder path: ").strip()
    
#     while True:
#         try:
#             blur_intensity = int(input("Enter blur intensity (odd number, e.g., 15, 25, 99): "))
#             if blur_intensity % 2 == 0:
#                 print("Please enter an odd number for blur intensity.")
#             else:
#                 break
#         except ValueError:
#             print("Invalid input. Please enter a valid odd number.")
    
#     while True:
#         try:
#             confidence_threshold = float(input("Enter face detection confidence level (0.0 to 1.0): "))
#             if 0.0 <= confidence_threshold <= 1.0:
#                 break
#             else:
#                 print("Please enter a number between 0.0 and 1.0.")
#         except ValueError:
#             print("Invalid input. Please enter a valid number between 0.0 and 1.0.")
    
#     blur_faces_in_folder(input_folder, output_folder, blur_intensity, confidence_threshold)

import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, Label, Entry, Button, IntVar, DoubleVar, Toplevel, messagebox

def get_fourcc(extension):
    if extension == 'mp4':
        return cv2.VideoWriter_fourcc(*'mp4v')
    elif extension == 'avi':
        return cv2.VideoWriter_fourcc(*'XVID')
    elif extension == 'mov':
        return cv2.VideoWriter_fourcc(*'mp4v')
    else:
        raise ValueError("Unsupported file format")

def blur_faces_in_video(input_file, output_folder, blur_intensity, confidence_threshold):
    # Load the DNN face detector model
    if not os.path.exists('deploy.prototxt') or not os.path.exists('res10_300x300_ssd_iter_140000_fp16.caffemodel'):
        messagebox.showerror("Error", "Face detection model files are missing!")
        return
    
    net = cv2.dnn.readNetFromCaffe(
        'deploy.prototxt', 
        'res10_300x300_ssd_iter_140000_fp16.caffemodel'
    )

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open video capture
    cap = cv2.VideoCapture(input_file)
    if not cap.isOpened():
        print(f"Error opening video file: {input_file}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Get file extension and define codec
    extension = input_file.split('.')[-1]
    fourcc = get_fourcc(extension)
    
    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + "_blurred." + extension)

    # Create VideoWriter object
    out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")

                # Ensure the bounding box is within image bounds
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w, x1), min(h, y1)

                if x < x1 and y < y1 and frame[y:y1, x:x1].size > 0:
                    face = frame[y:y1, x:x1]
                    blurred_face = cv2.GaussianBlur(face, (blur_intensity, blur_intensity), 30)
                    frame[y:y1, x:x1] = blurred_face

        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Processed video saved at: {output_file}")

# GUI for selecting parameters and running the program
def browse_input_file(entry):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mov")])
    entry.delete(0, "end")
    entry.insert(0, file_path)

def browse_output_folder(entry):
    folder_path = filedialog.askdirectory()
    entry.delete(0, "end")
    entry.insert(0, folder_path)

def start_processing(input_file_entry, output_folder_entry, blur_intensity_var, confidence_threshold_var):
    input_file = input_file_entry.get()
    output_folder = output_folder_entry.get()
    blur_intensity = blur_intensity_var.get()
    confidence_threshold = confidence_threshold_var.get()

    if not input_file or not os.path.exists(input_file):
        messagebox.showerror("Error", "Please select a valid input video file.")
        return

    if not output_folder or not os.path.exists(output_folder):
        messagebox.showerror("Error", "Please select a valid output folder.")
        return

    if blur_intensity % 2 == 0:
        messagebox.showerror("Error", "Blur intensity must be an odd number.")
        return

    blur_faces_in_video(input_file, output_folder, blur_intensity, confidence_threshold)
    messagebox.showinfo("Success", "Video processing complete!")

if __name__ == "__main__":
    root = Tk()
    root.title("Blur Faces in Video")

    # Input file selection
    Label(root, text="Input Video File:").grid(row=0, column=0, padx=10, pady=5, sticky="e")
    input_file_entry = Entry(root, width=50)
    input_file_entry.grid(row=0, column=1, padx=10, pady=5)
    Button(root, text="Browse", command=lambda: browse_input_file(input_file_entry)).grid(row=0, column=2, padx=10, pady=5)

    # Output folder selection
    Label(root, text="Output Folder:").grid(row=1, column=0, padx=10, pady=5, sticky="e")
    output_folder_entry = Entry(root, width=50)
    output_folder_entry.grid(row=1, column=1, padx=10, pady=5)
    Button(root, text="Browse", command=lambda: browse_output_folder(output_folder_entry)).grid(row=1, column=2, padx=10, pady=5)

    # Blur intensity
    Label(root, text="Blur Intensity (odd number):").grid(row=2, column=0, padx=10, pady=5, sticky="e")
    blur_intensity_var = IntVar(value=15)
    Entry(root, textvariable=blur_intensity_var).grid(row=2, column=1, padx=10, pady=5)

    # Confidence threshold
    Label(root, text="Confidence Threshold (0.0 - 1.0):").grid(row=3, column=0, padx=10, pady=5, sticky="e")
    confidence_threshold_var = DoubleVar(value=0.5)
    Entry(root, textvariable=confidence_threshold_var).grid(row=3, column=1, padx=10, pady=5)

    # Start button
    Button(root, text="Start Processing", command=lambda: start_processing(
        input_file_entry, output_folder_entry, blur_intensity_var, confidence_threshold_var
    )).grid(row=4, column=0, columnspan=3, pady=10)

    root.mainloop()

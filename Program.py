#GearO3D python3 program.py 17.03.2025
# Activate conda base environment
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import cv2
import numpy as np
import math
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os

import sys

import freecad
import Part
import freecad.gears.commands
import Mesh

# create the root window
root = tk.Tk()
root.title('GearO3D')
root.resizable(False, False)
root.geometry('300x300') 

def select_file():
    global filename
    global folder_selected
    
    folder_selected = fd.askdirectory()
    filetypes = (
        ('img files', '*.jpg'),
        ('All files', '*.*')
    )

    filename = fd.askopenfilename(
        title='Open a file',
        initialdir=folder_selected,
        filetypes=filetypes)

    showinfo(
        title='Selected File',
        message=filename
        
    )
    dosya_1 = filename
    print(filename)

def parametreler():
    # 1. Load the image
    global num_peaks
    global mod_
    img = cv2.imread(filename)  # your gear image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    # Closing small holes or removing noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find the largest contour (the gear boundary)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gear_contour = max(contours, key=cv2.contourArea)  # largest contour by area

    # Compute gear center (from contour moments)
    M = cv2.moments(gear_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        # fallback, e.g., boundingRect center
        x, y, w, h = cv2.boundingRect(gear_contour)
        cx, cy = x + w//2, y + h//2

    # Extract boundary points into a simpler format
    # gear_contour is an array of [[x, y]] points
    # we can measure angle and radius from the center for each point
    boundary_points = gear_contour.reshape(-1, 2)  # Nx2 array

    angles = []
    radii = []

    for (px, py) in boundary_points:
        dx = px - cx
        dy = py - cy
        angle = math.atan2(dy, dx)
        radius = math.hypot(dx, dy)
        angles.append(angle)
        radii.append(radius)

    # Convert angles to range [0, 2*pi)
    angles = np.array(angles)
    angles = np.mod(angles, 2*math.pi)
    radii = np.array(radii)

    # Sort by angle so we get a continuous angle -> radius mapping
    sort_idx = np.argsort(angles)
    angles_sorted = angles[sort_idx]
    radii_sorted = radii[sort_idx]

    # At this point, angles_sorted and radii_sorted define the boundary in polar coords.
    
    # Smooth the radius signal (optional but often helps)
    # You can use cv2.GaussianBlur on the radii array if needed:
    window_size = 21  # or some other odd number
    radii_smooth = cv2.GaussianBlur(radii_sorted.reshape(-1,1), (window_size,1), 0).flatten()
    # Moving average function for noise reduction
    
    cumsum = np.cumsum(np.insert(radii_smooth, 0, 0))
    cumsum_2 = np.cumsum(np.insert(angles_sorted, 0, 0))
    
    y_smooth = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)
    x_smooth = (cumsum_2[window_size:] - cumsum_2[:-window_size]) / float(window_size)
    
    #print(max(y_smooth))
    #print(min(y_smooth))
    dis_yukseligi_in_pixels = max(y_smooth)-min(y_smooth)
    #print(dis_yukseligi_in_pixels)

    sensor_width = 6.17 #Sensor size:28.0735mm2 (6.17mm x 4.55mm)
    image_width = 4320
    ppitch = sensor_width / image_width # mm/pixel
    distance_to_cam = 40
    focal_length = 6 #5-20 mm
    dis_yukseligi_in_mm = math.ceil((distance_to_cam / focal_length)*dis_yukseligi_in_pixels*ppitch)
    print(f"dis_yukseligi: {dis_yukseligi_in_mm}")

    # 8. Use peak-finding to count gear teeth
    # The find_peaks function in scipy.signal can detect local maxima.
    peaks, _ = find_peaks(y_smooth,  prominence=10, width=50)
    # 'distance' ensures minimum spacing between peaks, 
    # 'prominence' ensures peaks are sufficiently tall.

    num_peaks = len(peaks)
    print(f"Diş sayısı: {len(peaks)}")
    mod_ =dis_yukseligi_in_mm /2
    print(f"modulus: {mod_}")

    # If you need the peak heights (the 'highest' peaks), 
    #    you can look up their values in the radii_smooth array:
    peak_values = radii_smooth[peaks]

    # Visualize results

    plt.figure(figsize=(10,5))
    plt.plot(x_smooth, y_smooth, label="Radius (smoothed)")
    plt.plot(x_smooth[peaks], y_smooth[peaks], 'ro', label="Detected peaks")
    plt.xlabel("Angle (radians)")
    plt.ylabel("Radius")
    plt.legend()
    plt.show()
    # open button
    
def parametreler_2():
    global thickness_in_mm
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: The image did not load. Check the file path or file format.")
    else:
        # Apply a slight blur or smoothing if there's noise
        blurred = cv2.GaussianBlur(img, (3, 3), 0)

        # Threshold the image to separate the gear (white) from the background (black)
        # You might have to tweak the threshold value:
        _, thresh = cv2.threshold(blurred, 128, 255, cv2.THRESH_BINARY)

        # Find external contours in the thresholded image
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # If there are multiple contours, pick the largest one assuming it's the gear
        if len(contours) == 0:
            raise ValueError("No contours found. Check your thresholding or input image.")
        gear_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(gear_contour)

        # Depending on the gear orientation, the thickness might be w (width) or h (height).
        
        thickness_in_pixels = w
        print("Thickness in pixels (bounding rect):", thickness_in_pixels)

        sensor_width = 6.17 #Sensor size:28.0735mm2 (6.17mm x 4.55mm)
        image_width = 4320
        ppitch = sensor_width / image_width # mm/pixel
        distance_to_cam = 40
        focal_length = 6 #5-20 mm

        thickness_in_mm = math.ceil((distance_to_cam / focal_length)*thickness_in_pixels*ppitch)
        
        print("Yükseklik:", math.ceil(thickness_in_mm))

def stl():
    current_dir = os.getcwd()
    file_name_data = "data.txt"
    dosya_1 = os.path.join(current_dir, file_name_data)
    
    # Write to textfile
    with open(dosya_1, "w") as f:
        f.write(f"{int(num_peaks)}\n")  # Write first variable
        f.write(f"{int(mod_)}\n")  # Write second variable
        f.write(f"{int(thickness_in_mm)}\n")  # Write third variable
        f.write(f"{(folder_selected)}\n")  # Write third variable
    print("Data file created")
    
def freecad_():
    # Current working directory
    current_dir = os.getcwd()

    with open('data.txt', 'r') as file:
        lines = file.readlines()

    # lines is a list of strings, e.g. ["line1\n", "line2\n", "line3\n"]

    var1 = lines[0].strip()
    var2 = lines[1].strip()
    var3 = lines[2].strip()

    print(var1, var2, var3)
    doc = FreeCAD.newDocument("Test")

    gear = freecad.gears.commands.CreateInvoluteGear.create()

    gear.num_teeth = int(var1)
    gear.module = int(var2)
    gear.beta = 0
    gear.height = int(var3)
    gear.double_helix = False
    
    part_feature = doc.addObject("Part::Feature", "test")
    
    doc.recompute()

    file_name_fc = "gear.fcstd"
    save_path = os.path.join(current_dir, file_name_fc)
    # 4. Save the document (FCStd is the native FreeCAD format)
    doc.saveAs(save_path)

    print("Document saved successfully!")

    # Export.
    objects_for_export = [gear]
    # Export to STL

    file_name = "gear.stl"
    export_path = os.path.join(current_dir, file_name)
    Mesh.export(objects_for_export, export_path)

    print("Export complete:", export_path)
    
    print("Freecad files created")
    
open_button = ttk.Button(
    root,
    text='Open a File',
    command=select_file
)

parametreler_button = ttk.Button(
    root,
    text='Parametreler',
    command=parametreler
)

parametreler_2_button = ttk.Button(
    root,
    text='Parametreler_2',
    command=parametreler_2
)

stl_button = ttk.Button(
    root,
    text="stl",
    command=stl
)

freecad_button = ttk.Button(
    root,
    text='freecad',
    command=freecad_
)

open_button.pack(expand=True)
parametreler_button.pack(expand=True)
parametreler_2_button.pack(expand=True)
stl_button.pack(expand=True)
freecad_button.pack(expand=True)

# run the application
root.mainloop()

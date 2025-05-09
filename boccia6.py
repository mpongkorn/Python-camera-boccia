import cv2
import numpy as np
import math
from tkinter import *
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading
# from inference_sdk import InferenceHTTPClient
# from ultralytics import YOLO

import pymcprotocol

class FieldMeasureApp:
    # ---------- settings ----------
    PREVIEW_SCALE = 0.5
    LOUPE_SCALE = 2.0
    LOUPE_WIN = 200
    FIELD_W_CM = 400.0
    FIELD_H_CM = 500.0
    LOUPE_BORDER = 2
    LOUPE_BORDER_COLOR = (0, 255, 0)
    TARGET_COLOR = (0, 0, 255)
    TARGET_RADIUS = 2
    MIN_BALL_RADIUS = 12
    BALL_CIRCULARITY_THRESHOLD = 0.85

    BALL_COLORS = {
        0: ('Blue', (255, 0, 0)),
        1: ('White', (0, 0, 0)),
        2: ('Red', (0, 0, 255))
    }
    DETECTION_DRAW_COLORS = {
        'white': (0, 165, 255),
        'red': (0, 0, 255),
        'blue': (255, 0, 0),
        'default': (0, 255, 0)
    }
    # --------------------------------

    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("Field Measurement Tool")

        self.img = None
        self.preview = None
        self.cap = None
        self.camera_mode = False
        self.field_pts = []
        self.ball_pt1 = None
        self.ball_all = []
        self.cursor_preview = (0, 0)
        self.ball_detected = False
        self.running = True
        self.top_view_photo = None
        self.loupe_photo = None
        self.canvas_photo = None

        self.detection_threshold = 0.01

        self.target_x2_cm_str = StringVar(value="300.0")
        self.target_y2_cm_str = StringVar(value="750.0")

        # For HSV Color Picking
        self.picking_hsv_for_color = None
        self.default_color_ranges = {
            'white': [(np.array([0, 0, 150]), np.array([180, 65, 255]))], # Adjusted default for white
            'red': [ (np.array([0, 70, 70]), np.array([10, 255, 255])),   # More robust S,V defaults
                     (np.array([170, 70, 70]), np.array([179, 255, 255]))],
            'blue': [(np.array([100, 70, 70]), np.array([140, 255, 255]))] # More robust S,V defaults
        }
        self.active_color_ranges = self.default_color_ranges.copy()


        self.create_widgets()
        self.camera_thread = None
        self.min_radius_scale.set(self.MIN_BALL_RADIUS)
        self.circularity_scale.set(int(self.BALL_CIRCULARITY_THRESHOLD * 100))

    def create_widgets(self):
        main_frame = Frame(self.root)
        main_frame.pack(fill=BOTH, expand=True)

        self.image_frame = Frame(main_frame)
        self.image_frame.pack(side=LEFT, padx=10, pady=10, fill=BOTH, expand=True)

        self.canvas = Canvas(self.image_frame, background="lightgrey")
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Motion>", self.update_loupe_and_coords)
        self.canvas.bind("<Button-1>", self.handle_canvas_click)

        self.loupe_label = Label(self.image_frame, background="lightgrey")
        self.loupe_label.pack(pady=5)

        self.coord_label = Label(self.image_frame, text="X: -, Y: -")
        self.coord_label.pack(pady=5)

        self.control_frame = Frame(main_frame, width=320)
        self.control_frame.pack(side=RIGHT, padx=10, pady=10, fill=Y)
        self.control_frame.pack_propagate(False)

        input_mode_frame = LabelFrame(self.control_frame, text="Input Mode", font=('Arial', 10, 'bold'))
        input_mode_frame.pack(fill=X, padx=5, pady=5)
        Button(input_mode_frame, text="Open Image File", command=self.open_image_file).pack(fill=X, pady=2)
        Button(input_mode_frame, text="Open Camera", command=self.open_camera).pack(fill=X, pady=2)
        self.capture_btn = Button(input_mode_frame, text="Capture Frame", command=self.capture_frame, state=DISABLED)
        self.capture_btn.pack(fill=X, pady=2)

        corners_frame = LabelFrame(self.control_frame, text="Field Corners", font=('Arial', 10, 'bold'))
        corners_frame.pack(fill=X, padx=5, pady=5)
        self.corner_listbox = Listbox(corners_frame, height=4, width=30)
        self.corner_listbox.pack(fill=X, pady=2)
        corner_buttons_frame = Frame(corners_frame)
        corner_buttons_frame.pack(fill=X)
        Button(corner_buttons_frame, text="Remove Last", command=self.remove_last_point).pack(side=LEFT, expand=True, fill=X, pady=2, padx=1)
        Button(corner_buttons_frame, text="Clear All", command=self.clear_points).pack(side=LEFT, expand=True, fill=X, pady=2, padx=1)

        ball_detect_frame = LabelFrame(self.control_frame, text="Ball Detection", font=('Arial', 10, 'bold'))
        ball_detect_frame.pack(fill=X, padx=5, pady=5)
        Button(ball_detect_frame, text="Detect Balls (Manual)", command=self.run_manual_ball_detection).pack(fill=X, pady=2)
        self.ball_status = Label(ball_detect_frame, text="Primary Ball: Not detected")
        self.ball_status.pack(fill=X, pady=2)

        params_frame = LabelFrame(self.control_frame, text="Primary Ball Detection Params", font=('Arial', 10, 'bold'))
        params_frame.pack(fill=X, padx=5, pady=5)
        Label(params_frame, text="Min Radius (px):").grid(row=0, column=0, sticky=W)
        self.min_radius_scale = Scale(params_frame, from_=1, to=50, orient=HORIZONTAL, command=self.update_detection_params)
        self.min_radius_scale.grid(row=0, column=1, sticky='ew')
        Label(params_frame, text="Circularity (%):").grid(row=1, column=0, sticky=W)
        self.circularity_scale = Scale(params_frame, from_=0, to=100, orient=HORIZONTAL, command=self.update_detection_params)
        self.circularity_scale.grid(row=1, column=1, sticky='ew')
        params_frame.grid_columnconfigure(1, weight=1)

        # MODIFICATION: HSV Color Calibration Section
        color_calib_frame = LabelFrame(self.control_frame, text="HSV Color Calibration", font=('Arial', 10, 'bold'))
        color_calib_frame.pack(fill=X, padx=5, pady=5)
        Button(color_calib_frame, text="Pick White Ball Color", command=lambda: self.initiate_hsv_color_pick('white')).pack(fill=X, pady=2)
        Button(color_calib_frame, text="Pick Red Ball Color", command=lambda: self.initiate_hsv_color_pick('red')).pack(fill=X, pady=2)
        Button(color_calib_frame, text="Pick Blue Ball Color", command=lambda: self.initiate_hsv_color_pick('blue')).pack(fill=X, pady=2)
        Button(color_calib_frame, text="Reset Colors to Default", command=self.reset_hsv_colors_to_default).pack(fill=X, pady=2)


        target_ball_frame = LabelFrame(self.control_frame, text="Target Ball 2 (Real World CM)", font=('Arial', 10, 'bold'))
        target_ball_frame.pack(fill=X, padx=5, pady=5)
        Label(target_ball_frame, text="X2 (cm):").grid(row=0, column=0, sticky=W, padx=2, pady=2)
        self.x2_entry = Entry(target_ball_frame, textvariable=self.target_x2_cm_str, width=10)
        self.x2_entry.grid(row=0, column=1, sticky='ew', padx=2, pady=2)
        Label(target_ball_frame, text="Y2 (cm):").grid(row=1, column=0, sticky=W, padx=2, pady=2)
        self.y2_entry = Entry(target_ball_frame, textvariable=self.target_y2_cm_str, width=10)
        self.y2_entry.grid(row=1, column=1, sticky='ew', padx=2, pady=2)
        target_ball_frame.grid_columnconfigure(1, weight=1)

        process_frame = LabelFrame(self.control_frame, text="Measurement", font=('Arial', 10, 'bold'))
        process_frame.pack(fill=X, padx=5, pady=5)
        Button(process_frame, text="Calculate Measurements", command=self.process_measurements,
               bg='lightblue', font=('Arial', 10, 'bold')).pack(fill=X, pady=5)

    def initiate_hsv_color_pick(self, color_name):
        if self.img is None:
            print(f"Warning: Please open an image or camera first to pick {color_name} color.")
            return

        if self.picking_hsv_for_color is not None and self.picking_hsv_for_color != color_name :
            print(f"Info: Color picking for {self.picking_hsv_for_color} cancelled. Now picking for {color_name}.")
        elif self.picking_hsv_for_color == color_name: # Clicked same button again to cancel
            print(f"Info: Color picking for {color_name} cancelled by user.")
            self.canvas.config(cursor="")
            self.picking_hsv_for_color = None
            return

        self.picking_hsv_for_color = color_name
        self.canvas.config(cursor="crosshair")
        print(f"Info: Click on a {color_name} ball in the image to set its HSV range. Click button again to cancel.")

    def reset_hsv_colors_to_default(self):
        self.active_color_ranges = self.default_color_ranges.copy()
        print("Info: HSV color ranges reset to defaults.")
        if self.img is not None and len(self.field_pts) == 4:
            self.run_manual_ball_detection()

    def process_hsv_color_pick(self, event):
        if self.preview is None or self.picking_hsv_for_color is None:
            if self.picking_hsv_for_color:
                 print(f"Warning: HSV Color picking for {self.picking_hsv_for_color} failed or was interrupted.")
            self.canvas.config(cursor="")
            self.picking_hsv_for_color = None
            return

        color_name_being_picked = self.picking_hsv_for_color
        X_orig = int(event.x / self.PREVIEW_SCALE)
        Y_orig = int(event.y / self.PREVIEW_SCALE)

        if 0 <= X_orig < self.img.shape[1] and 0 <= Y_orig < self.img.shape[0]:
            # Extract a small patch (e.g., 5x5) and average its HSV
            patch_size = 5
            half_patch = patch_size // 2
            y_start = max(0, Y_orig - half_patch)
            y_end = min(self.img.shape[0], Y_orig + half_patch + 1)
            x_start = max(0, X_orig - half_patch)
            x_end = min(self.img.shape[1], X_orig + half_patch + 1)

            bgr_patch = self.img[y_start:y_end, x_start:x_end]
            if bgr_patch.size == 0:
                print(f"Warning: Empty patch for HSV color pick for {color_name_being_picked}.")
                self.canvas.config(cursor="")
                self.picking_hsv_for_color = None
                return

            hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
            # Calculate median HSV for robustness against outliers in patch
            h_picked = int(np.median(hsv_patch[:,:,0]))
            s_picked = int(np.median(hsv_patch[:,:,1]))
            v_picked = int(np.median(hsv_patch[:,:,2]))

            print(f"Info: Picked BGR (center of patch): {self.img[Y_orig, X_orig]}, Median HSV of patch: [{h_picked}, {s_picked}, {v_picked}] for {color_name_being_picked}")

            h_delta = 10
            s_delta = 60
            v_delta = 60
            s_min_threshold = 50
            v_min_threshold = 50

            lower_h = max(0, h_picked - h_delta)
            upper_h = min(179, h_picked + h_delta)
            lower_s = max(s_min_threshold, s_picked - s_delta)
            upper_s = min(255, s_picked + s_delta)
            lower_v = max(v_min_threshold, v_picked - v_delta)
            upper_v = min(255, v_picked + v_delta)

            picked_hsv_lower = np.array([lower_h, lower_s, lower_v])
            picked_hsv_upper = np.array([upper_h, upper_s, upper_v])

            if color_name_being_picked == 'red' and (h_picked < h_delta + 5 or h_picked > 179 - (h_delta + 5)):
                 print(f"Info: Picked red color (H:{h_picked}) is near Hue boundaries. Using default dual range for red for robustness.")
                 self.active_color_ranges[color_name_being_picked] = list(self.default_color_ranges[color_name_being_picked]) # Ensure it's a list of tuples
            else:
                self.active_color_ranges[color_name_being_picked] = [(picked_hsv_lower, picked_hsv_upper)]

            print(f"Info: Updated HSV range for {color_name_being_picked}: {self.active_color_ranges[color_name_being_picked]}")
        else:
            print(f"Warning: HSV Color pick for {color_name_being_picked} was outside image bounds.")

        self.canvas.config(cursor="")
        self.picking_hsv_for_color = None
        if self.img is not None and len(self.field_pts) == 4:
            self.run_manual_ball_detection()

    def update_loupe_and_coords(self, event):
        if self.img is None or self.preview is None:
            return
        preview_h, preview_w = self.preview.shape[:2]
        if 0 <= event.x < preview_w and 0 <= event.y < preview_h:
            self.cursor_preview = (event.x, event.y)
            self.coord_label.config(text=f"X: {event.x}, Y: {event.y} (Preview)")
            X_orig = int(event.x / self.PREVIEW_SCALE)
            Y_orig = int(event.y / self.PREVIEW_SCALE)
            half_loupe_orig = int(self.LOUPE_WIN / (2 * self.LOUPE_SCALE))
            x1 = max(0, X_orig - half_loupe_orig)
            y1 = max(0, Y_orig - half_loupe_orig)
            x2 = min(self.img.shape[1], X_orig + half_loupe_orig)
            y2 = min(self.img.shape[0], Y_orig + half_loupe_orig)
            patch = self.img[y1:y2, x1:x2]
            if patch.size == 0:
                patch = np.zeros((self.LOUPE_WIN // int(self.LOUPE_SCALE), self.LOUPE_WIN // int(self.LOUPE_SCALE), 3), dtype=np.uint8)
            loupe_img = cv2.resize(patch, (self.LOUPE_WIN, self.LOUPE_WIN), interpolation=cv2.INTER_NEAREST)
            center_loupe = self.LOUPE_WIN // 2
            cv2.circle(loupe_img, (center_loupe, center_loupe), self.TARGET_RADIUS, self.TARGET_COLOR, -1)
            cv2.rectangle(loupe_img, (0, 0), (self.LOUPE_WIN - 1, self.LOUPE_WIN - 1), self.LOUPE_BORDER_COLOR, self.LOUPE_BORDER)
            loupe_rgb = cv2.cvtColor(loupe_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(loupe_rgb)
            self.loupe_photo = ImageTk.PhotoImage(image=img_pil)
            self.loupe_label.config(image=self.loupe_photo, width=self.LOUPE_WIN, height=self.LOUPE_WIN)
        else:
            self.coord_label.config(text="X: -, Y: - (Outside Preview)")

    def handle_canvas_click(self, event):
        if self.picking_hsv_for_color is not None:
            self.process_hsv_color_pick(event)
        else:
            self.add_point(event)

    def open_image_file(self):
        self.stop_camera_if_running()
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            try:
                self.img = cv2.imread(file_path)
                if self.img is None:
                    raise ValueError(f"Could not read image file: {file_path}")
                self.canvas.update_idletasks()
                canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
                img_h, img_w = self.img.shape[:2]
                scale_w = canvas_w / img_w
                scale_h = canvas_h / img_h
                self.PREVIEW_SCALE = min(scale_w, scale_h, 1.0)
                self.preview = cv2.resize(
                    self.img, None, fx=self.PREVIEW_SCALE, fy=self.PREVIEW_SCALE,
                    interpolation=cv2.INTER_AREA)
                self.reset_state_for_new_image()
                self.update_main_canvas_display()
            except Exception as e:
                print(f"Error: Could not open image: {str(e)}")

    def reset_state_for_new_image(self):
        self.field_pts = []
        self.ball_pt1 = None
        self.ball_all = []
        self.ball_detected = False
        self.ball_status.config(text="Primary Ball: Not detected")
        self.corner_listbox.delete(0, END)
        if hasattr(self, 'capture_btn'):
            self.capture_btn.config(state=DISABLED)
        self.camera_mode = False
        # Reset active colors to default when a new image is loaded
        self.active_color_ranges = self.default_color_ranges.copy()
        print("Info: Active HSV color ranges reset to defaults for new image.")


    def open_camera(self):
        self.stop_camera_if_running()
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(1)
                if not self.cap.isOpened():
                    raise ValueError("Could not open camera. Check connection and permissions.")
            self.camera_mode = True
            self.capture_btn.config(state=NORMAL)
            self.reset_state_for_new_image() # This will also reset colors to default
            self.camera_thread = threading.Thread(target=self.update_camera_feed, daemon=True)
            self.camera_thread.start()
        except Exception as e:
            print(f"Error: Could not open camera: {str(e)}")
            if self.cap: self.cap.release()
            self.cap = None

    def stop_camera_if_running(self):
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.running = True
        self.camera_mode = False

    def update_camera_feed(self):
        while self.cap is not None and self.cap.isOpened() and self.running and self.camera_mode:
            ret, frame = self.cap.read()
            if ret:
                self.img = frame.copy()
                self.canvas.update_idletasks()
                canvas_w, canvas_h = self.canvas.winfo_width(), self.canvas.winfo_height()
                if canvas_w > 1 and canvas_h > 1 and self.img is not None:
                    img_h, img_w = self.img.shape[:2]
                    if img_w > 0 and img_h > 0:
                        scale_w = canvas_w / img_w
                        scale_h = canvas_h / img_h
                        self.PREVIEW_SCALE = min(scale_w, scale_h, 0.75)
                        self.preview = cv2.resize(
                            self.img, None, fx=self.PREVIEW_SCALE, fy=self.PREVIEW_SCALE,
                            interpolation=cv2.INTER_AREA)
                        self.update_main_canvas_display()
            else:
                print("Warning: Could not read frame from camera.")
                pass

    def detect_balls_in_frame(self):
        if self.img is None:
            self.ball_pt1 = None
            self.ball_all = []
            self.ball_detected = False
            print("Info: No image loaded for ball detection.")
            return

        if len(self.field_pts) != 4:
            print("Info: Ball detection requires 4 field corners to be selected first.")
            self.ball_pt1 = None
            self.ball_all = []
            self.ball_detected = False
            self.ball_status.config(text="Primary Ball: Select 4 corners")
            self.update_main_canvas_display()
            return

        field_mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        try:
            pts_for_poly = np.array([self.field_pts[0], self.field_pts[1], self.field_pts[3], self.field_pts[2]], dtype=np.int32)
            cv2.fillPoly(field_mask, [pts_for_poly], 255)
            masked_img_for_detection = cv2.bitwise_and(self.img, self.img, mask=field_mask)
        except IndexError:
            print("Error: Field points index error during mask creation.")
            masked_img_for_detection = self.img.copy()

        self.ball_pt1 = None
        self.ball_detected = False
        self.ball_status.config(text="Primary Ball: Not detected")

        lab = cv2.cvtColor(masked_img_for_detection, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge((l_channel, a_channel, b_channel))
        enhanced_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced_img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Use self.active_color_ranges which can be updated by user
        color_ranges_to_use = self.active_color_ranges

        all_detected_objects = []
        best_hsv_white_ball_for_primary = None
        highest_primary_white_metric = 0.0
        current_min_radius_for_primary = self.MIN_BALL_RADIUS
        current_circularity_thresh_for_primary = self.BALL_CIRCULARITY_THRESHOLD

        for color_name, hsv_ranges_list in color_ranges_to_use.items(): # MODIFIED
            combined_color_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower_hsv, upper_hsv in hsv_ranges_list:
                individual_mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                combined_color_mask = cv2.bitwise_or(combined_color_mask, individual_mask)
            kernel = np.ones((5, 5), np.uint8)
            if color_name == 'white':
                morph_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
                morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            else:
                morph_mask = cv2.morphologyEx(combined_color_mask, cv2.MORPH_OPEN, kernel, iterations=1)
                morph_mask = cv2.dilate(morph_mask, kernel, iterations=1)
                morph_mask = cv2.morphologyEx(morph_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
            contours_hsv, _ = cv2.findContours(morph_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours_hsv:
                area = cv2.contourArea(cnt)
                if 200 < area < 50000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0: continue
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    (x,y), radius = cv2.minEnclosingCircle(cnt)
                    if radius > current_min_radius_for_primary * 0.8 and circularity > current_circularity_thresh_for_primary * 0.8: # General filter
                        hull = cv2.convexHull(cnt)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / hull_area if hull_area > 0 else 0
                        if solidity > 0.75:
                            class_id = -1
                            if color_name == 'white': class_id = 1
                            elif color_name == 'red': class_id = 2
                            elif color_name == 'blue': class_id = 0
                            current_detected_ball = {
                                'bbox': (int(x-radius), int(y-radius), int(x+radius), int(y+radius)),
                                'confidence': (circularity * 0.6 + solidity * 0.4),
                                'class': class_id, 'center': (int(x), int(y)),
                                'radius': int(radius), 'color_name': color_name,
                                'circularity': circularity
                            }
                            all_detected_objects.append(current_detected_ball)
                            if color_name == 'white': # Strict check for primary
                                if circularity > highest_primary_white_metric and \
                                   radius >= current_min_radius_for_primary and \
                                   circularity >= current_circularity_thresh_for_primary:
                                    highest_primary_white_metric = circularity
                                    best_hsv_white_ball_for_primary = current_detected_ball
        if best_hsv_white_ball_for_primary:
            self.ball_pt1 = best_hsv_white_ball_for_primary['center']
            self.ball_detected = True
            self.ball_status.config(text=f"Primary Ball (HSV): Found at {self.ball_pt1} (Circ: {highest_primary_white_metric:.2f})")

        boxes_for_nms = []
        confidences_for_nms = []
        indices_orig = []
        for i, det in enumerate(all_detected_objects):
            x1, y1, x2, y2 = det['bbox']
            boxes_for_nms.append([x1, y1, x2-x1, y2-y1])
            confidences_for_nms.append(det['confidence'])
            indices_orig.append(i)
        if len(boxes_for_nms) > 0:
            nms_indices = cv2.dnn.NMSBoxes(boxes_for_nms, np.array(confidences_for_nms),
                                           score_threshold=self.detection_threshold, nms_threshold=0.4)
            if isinstance(nms_indices, np.ndarray):
                nms_indices = nms_indices.flatten()
            self.ball_all = [all_detected_objects[indices_orig[i]] for i in nms_indices]
        else:
            self.ball_all = []
        return

    def capture_frame(self):
        if self.camera_mode and self.img is not None:
            self.capture_btn.config(state=DISABLED)
            print("Info: Current camera frame is locked for measurement.")
            self.camera_mode = False
            self.update_main_canvas_display()
        elif not self.camera_mode and self.img is not None:
            print("Info: Already using a static image or captured frame.")
        else:
            print("Warning: Camera not active or no image loaded to capture.")

    def update_main_canvas_display(self):
        if self.preview is None:
            self.canvas.delete("all")
            self.canvas.update_idletasks()
            cv_w, cv_h = self.canvas.winfo_width(), self.canvas.winfo_height()
            if cv_w > 1 and cv_h > 1:
                self.canvas.create_text(cv_w//2, cv_h//2, text="No image loaded", font=("Arial", 16))
            return
        preview_to_draw_on = self.preview.copy()
        for i, pt_orig_coords in enumerate(self.field_pts):
            x_prev, y_prev = int(pt_orig_coords[0] * self.PREVIEW_SCALE), int(pt_orig_coords[1] * self.PREVIEW_SCALE)
            cv2.circle(preview_to_draw_on, (x_prev, y_prev), 7, (0, 255, 0), -1)
            cv2.putText(preview_to_draw_on, str(i + 1), (x_prev + 10, y_prev - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        if len(self.ball_all) > 0:
            for det_ball in self.ball_all:
                center_orig, radius_orig = det_ball['center'], det_ball['radius']
                color_name, confidence = det_ball['color_name'], det_ball['confidence']
                center_prev = (int(center_orig[0] * self.PREVIEW_SCALE), int(center_orig[1] * self.PREVIEW_SCALE))
                radius_prev = int(radius_orig * self.PREVIEW_SCALE)
                if radius_prev < 1: radius_prev = 1
                draw_color = self.DETECTION_DRAW_COLORS.get(color_name, self.DETECTION_DRAW_COLORS['default'])
                cv2.circle(preview_to_draw_on, center_prev, radius_prev, draw_color, 2)
                cv2.putText(preview_to_draw_on, f"{color_name[:1].upper()}{color_name[1:]} ({confidence:.2f})",
                            (center_prev[0] - radius_prev, center_prev[1] - radius_prev - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, draw_color, 1, cv2.LINE_AA)
        if self.ball_detected and self.ball_pt1:
            x_ball_orig, y_ball_orig = self.ball_pt1
            x_ball_prev, y_ball_prev = int(x_ball_orig * self.PREVIEW_SCALE), int(y_ball_orig * self.PREVIEW_SCALE)
            cv2.drawMarker(preview_to_draw_on, (x_ball_prev, y_ball_prev), (255,0,255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
            cv2.putText(preview_to_draw_on, "Primary", (x_ball_prev + 10, y_ball_prev + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2, cv2.LINE_AA)
        preview_rgb = cv2.cvtColor(preview_to_draw_on, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(preview_rgb)
        self.canvas_photo = ImageTk.PhotoImage(image=img_pil)
        self.canvas.config(width=img_pil.width, height=img_pil.height)
        self.canvas.create_image(0, 0, anchor=NW, image=self.canvas_photo)

    def update_detection_params(self, event=None):
        self.MIN_BALL_RADIUS = self.min_radius_scale.get()
        self.BALL_CIRCULARITY_THRESHOLD = self.circularity_scale.get() / 100.0
        if self.img is not None and not self.camera_mode and len(self.field_pts) == 4:
            self.run_manual_ball_detection()

    def add_point(self, event):
        if self.img is None or self.preview is None:
            print("Warning: No image loaded to select points on.")
            return
        preview_h, preview_w = self.preview.shape[:2]
        if not (0 <= event.x < preview_w and 0 <= event.y < preview_h):
            print("Warning: Clicked outside the image preview area.")
            return
        if len(self.field_pts) < 4:
            X_orig = int(event.x / self.PREVIEW_SCALE)
            Y_orig = int(event.y / self.PREVIEW_SCALE)
            self.field_pts.append((X_orig, Y_orig))
            self.corner_listbox.insert(END, f"Corner {len(self.field_pts)}: ({X_orig}, {Y_orig})")
            self.update_main_canvas_display()
            if len(self.field_pts) == 4:
                print("Info: All 4 field corners selected. You can now detect balls and process measurements.")
                if self.img is not None:
                    self.run_manual_ball_detection()
        else:
            print("Info: Already selected 4 corners. Clear points to re-select.")

    def save_quadrilateral(self):
        if len(self.field_pts) != 4 or self.img is None:
            print("Error: Need exactly 4 points and an image to save quadrilateral.")
            return
        mask = np.zeros(self.img.shape[:2], dtype=np.uint8)
        try:
            pts_np = np.array([self.field_pts[0], self.field_pts[1], self.field_pts[3], self.field_pts[2]], dtype=np.int32)
            cv2.fillPoly(mask, [pts_np], 255)
        except Exception as e:
            print(f"Error: Could not create polygon mask: {e}")
            return
        result_img = cv2.bitwise_and(self.img, self.img, mask=mask)
        x, y, w, h = cv2.boundingRect(pts_np)
        cropped_result = result_img[y:y+h, x:x+w]
        if cropped_result.size > 0 :
            if self.img.shape[2] == 3: # BGR
                b, g, r = cv2.split(cropped_result)
                alpha_channel = mask[y:y+h, x:x+w]
                transparent_img = cv2.merge((b, g, r, alpha_channel))
                try:
                    save_path_transparent = filedialog.asksaveasfilename(
                        defaultextension=".png", filetypes=[("PNG files", "*.png")],
                        title="Save Quadrilateral (Transparent)")
                    if save_path_transparent:
                        cv2.imwrite(save_path_transparent, transparent_img)
                        print(f"Info: Saved transparent quadrilateral to {save_path_transparent}")
                except Exception as e:
                    print(f"Error: Could not save transparent image: {e}")
            try:
                save_path_opaque = filedialog.asksaveasfilename(
                    defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg")],
                    title="Save Quadrilateral (Opaque)")
                if save_path_opaque:
                    cv2.imwrite(save_path_opaque, cropped_result)
                    print(f"Info: Saved opaque quadrilateral to {save_path_opaque}")
            except Exception as e:
                print(f"Error: Could not save opaque image: {e}")
        else:
            print("Error: Cropped quadrilateral area is empty.")

    def remove_last_point(self):
        if self.field_pts:
            removed_pt = self.field_pts.pop()
            self.corner_listbox.delete(END)
            print(f"Info: Removed point: {removed_pt}")
            self.update_main_canvas_display()
            if len(self.field_pts) < 4:
                self.ball_pt1 = None
                self.ball_all = []
                self.ball_detected = False
                self.ball_status.config(text="Primary Ball: Select 4 corners")
        else:
            print("Info: No points to remove.")

    def clear_points(self):
        self.field_pts = []
        self.corner_listbox.delete(0, END)
        print("Info: Cleared all field points.")
        self.ball_pt1 = None
        self.ball_all = []
        self.ball_detected = False
        self.ball_status.config(text="Primary Ball: Select 4 corners")
        self.update_main_canvas_display()

    def run_manual_ball_detection(self):
        if self.img is None:
            print("Warning: Please open an image or start camera first to detect balls.")
            return
        print("Info: Running manual ball detection...")
        self.detect_balls_in_frame()
        self.update_main_canvas_display()
        if len(self.field_pts) == 4: # Only print detection status if corners are set
            if not self.ball_all and not self.ball_pt1:
                print("Info: No balls detected with current settings.")
            elif self.ball_pt1: # This implies primary was found
                print(f"Info: Primary white ball detected at {self.ball_pt1}. Other balls: {len(self.ball_all)} found.")
            else: # No primary, but other balls might be present
                print(f"Info: Detected {len(self.ball_all)} balls (no primary white ball).")


    def process_measurements(self):
        if self.img is None:
            print("Warning: Please open an image or capture a frame first for measurement.")
            return
        if len(self.field_pts) != 4:
            print("Warning: Please mark all 4 field corners first for measurement.")
            return
        if not self.camera_mode :
             self.detect_balls_in_frame()
        if not self.ball_detected or self.ball_pt1 is None:
            print("Warning: Primary white ball not detected. Cannot perform measurement.")
            return

        src_pts = np.float32(self.field_pts)
        dst_pts = np.float32([[0, 0], [0, self.FIELD_H_CM], [self.FIELD_W_CM, 0], [self.FIELD_W_CM, self.FIELD_H_CM]])
        try:
            H_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        except Exception as e:
            print(f"Error: Could not compute perspective transform: {e}. Ensure points are not collinear and in correct order (TL, BL, TR, BR).")
            return

        ball1_original_img_pt_offset = (self.ball_pt1[0], self.ball_pt1[1] + 5)
        ball1_pixels_np = np.float32([ball1_original_img_pt_offset]).reshape(-1, 1, 2)
        ball1_cm_transformed = cv2.perspectiveTransform(ball1_pixels_np, H_matrix)
        if ball1_cm_transformed is None or ball1_cm_transformed.size == 0:
            print("Error: Could not transform ball coordinates.")
            return

        x1_cm, y1_cm = ball1_cm_transformed[0, 0]

        try:
            x2_cm_str = self.target_x2_cm_str.get()
            y2_cm_str = self.target_y2_cm_str.get()
            x2_cm = float(x2_cm_str)
            y2_cm = float(y2_cm_str)
            print(f"Info: Using Target Ball 2 position: X={x2_cm} cm, Y={y2_cm} cm")
        except ValueError:
            print(f"Warning: Invalid input for Target Ball 2 coordinates ('{x2_cm_str}', '{y2_cm_str}'). Using default 0.0, 0.0.")
            x2_cm = 0.0
            y2_cm = 0.0
            self.target_x2_cm_str.set(str(x2_cm))
            self.target_y2_cm_str.set(str(y2_cm))

        delta_x_cm = x2_cm - x1_cm
        delta_y_cm = y2_cm - y1_cm
        distance_cm = math.sqrt(delta_x_cm**2 + delta_y_cm**2)
        angle_radians = math.atan2(delta_x_cm, delta_y_cm)
        angle_degrees = math.degrees(angle_radians)
        if angle_degrees < 0: angle_degrees += 360.0

        warp_w_px, warp_h_px = int(self.FIELD_W_CM), int(self.FIELD_H_CM)
        display_scale_factor = 1.0
        display_warp_w = int(warp_w_px * display_scale_factor)
        display_warp_h = int(warp_h_px * display_scale_factor)
        if display_warp_w <= 0: display_warp_w = 100
        if display_warp_h <= 0: display_warp_h = 100

        warped_img_native_cm = cv2.warpPerspective(self.img, H_matrix, (warp_w_px, warp_h_px))
        cv2.circle(warped_img_native_cm, (int(x1_cm), int(y1_cm)), 6, (0, 0, 255), -1)
        cv2.putText(warped_img_native_cm, "B1", (int(x1_cm)+5, int(y1_cm)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.circle(warped_img_native_cm, (int(x2_cm), int(y2_cm)), 6, (0, 255, 0), -1)
        cv2.putText(warped_img_native_cm, "B2", (int(x2_cm)+5, int(y2_cm)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)
        cv2.arrowedLine(warped_img_native_cm, (int(x1_cm), int(y1_cm)), (int(x2_cm), int(y2_cm)), (255, 255, 255), 2)
        warped_img_display = cv2.resize(warped_img_native_cm, (display_warp_w, display_warp_h), interpolation=cv2.INTER_LINEAR)

        result_window = Toplevel(self.root)
        result_window.title("Measurement Results & Top View")
        warped_rgb_display = cv2.cvtColor(warped_img_display, cv2.COLOR_BGR2RGB)
        img_pil_display = Image.fromarray(warped_rgb_display)
        self.top_view_photo = ImageTk.PhotoImage(image=img_pil_display)
        Label(result_window, text="Field Top View (Scaled for display)", font=('Arial', 12)).pack(pady=5)
        warped_label = Label(result_window, image=self.top_view_photo)
        warped_label.pack(pady=10)
        results_frame = Frame(result_window)
        results_frame.pack(pady=10, padx=10, fill=X)
        Label(results_frame, text=f"Ball 1 - Original Image Coords (pixels): {self.ball_pt1}", font=('Arial', 10)).pack(anchor=W)
        Label(results_frame, text=f"Ball 1 - Field Coords (cm, from Top-Left): ({x1_cm:.1f}, {y1_cm:.1f})", font=('Arial', 10, 'bold')).pack(anchor=W)
        Label(results_frame, text=f"Target Ball 2 - Field Coords (cm, from Top-Left): ({x2_cm:.1f}, {y2_cm:.1f})", font=('Arial', 10)).pack(anchor=W)
        ttk.Separator(results_frame, orient='horizontal').pack(fill='x', pady=5)
        Label(results_frame, text=f"Distance B1 to B2: {distance_cm:.1f} cm", font=('Arial', 12, 'bold')).pack(anchor=W)
        Label(results_frame, text=f"Angle B1 to B2 (from B1, rel. to Y-axis): {angle_degrees:.1f}°", font=('Arial', 12, 'bold')).pack(anchor=W)
        Button(result_window, text="Close", command=result_window.destroy).pack(pady=10)
        result_window.grab_set()

    def on_closing(self):
        print("Closing application...")
        self.running = False
        self.stop_camera_if_running()
        cv2.destroyAllWindows()
        self.root.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = FieldMeasureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.minsize(1000, 700)
    root.mainloop()

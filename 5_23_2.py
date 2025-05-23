import cv2
import numpy as np
import math
from tkinter import *
from tkinter import filedialog, ttk, messagebox, StringVar, IntVar
from PIL import Image, ImageTk, ImageColor # Added ImageColor for safety, though not strictly needed for "lightgrey"
import threading
import traceback
import time
import copy
import pymcprotocol
import sys

# --- CalibrationWindow Class (Code from previous response) ---
class CalibrationWindow(Toplevel):
    def __init__(self, main_app):
        super().__init__(main_app.root)
        self.main_app = main_app
        self.title("Advanced Calibration")
        self.geometry("850x900")  # Adjusted to ensure it fits smaller screens if opened
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.calib_params = copy.deepcopy(self.main_app.active_detection_params)
        self.img_to_process = None
        if self.main_app.img is not None:
            self.img_to_process = self.main_app.img.copy()

        self.preview_size = (480, 320) # Slightly smaller preview to save space
        self.color_vars = {}

        self.create_widgets()
        self.load_params_to_ui()
        self.refresh_all_mask_displays()

    def create_widgets(self):
        main_frame = Frame(self)
        main_frame.pack(fill=BOTH, expand=True, padx=5, pady=5)

        global_settings_frame = LabelFrame(
            main_frame, text="Global Detection Settings", padx=5, pady=5
        )
        global_settings_frame.pack(fill=X, pady=(0, 5))

        Label(global_settings_frame, text="Gaussian Blur Kernel (odd, 3-21):").grid(
            row=0, column=0, sticky=W, padx=2, pady=2
        )
        self.color_vars["blur_kernel_scale"] = Scale(
            global_settings_frame,
            from_=3,
            to=21,
            orient=HORIZONTAL,
            resolution=2,
            command=lambda x: self.update_param_and_refresh_all("blur_kernel", int(x)),
        )
        self.color_vars["blur_kernel_scale"].grid(
            row=0, column=1, sticky=EW, padx=2, pady=2
        )
        global_settings_frame.grid_columnconfigure(1, weight=1)

        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=BOTH, expand=True, pady=3)

        colors_to_calibrate = ["white", "red", "blue"]
        for color_name in colors_to_calibrate:
            color_tab_frame = Frame(self.notebook)
            self.notebook.add(color_tab_frame, text=color_name.capitalize())
            self.setup_color_controls(color_tab_frame, color_name)

        action_frame = Frame(main_frame)
        action_frame.pack(fill=X, pady=5)
        Button(
            action_frame,
            text="Apply to Main",
            command=self.apply_changes,
            bg="lightgreen",
        ).pack(side=LEFT, padx=3)

        Button(
            action_frame,
            text="Reset Current Tab",
            command=self.reset_current_tab_to_defaults,
        ).pack(side=LEFT, padx=3)
        Button(
            action_frame,
            text="Refresh Previews",
            command=self.refresh_all_mask_displays_from_button,
        ).pack(side=LEFT, padx=3)
        Button(action_frame, text="Close", command=self.on_closing, bg="salmon").pack(
            side=RIGHT, padx=3
        )

    def setup_color_controls(self, parent_frame, color_name):
        left_panel = Frame(parent_frame)
        left_panel.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 3))

        right_panel = Frame(parent_frame, width=self.preview_size[0] + 10) # Adjusted width
        right_panel.pack(side=RIGHT, fill=Y, padx=(3, 0))
        right_panel.pack_propagate(False)

        hsv_frame = LabelFrame(left_panel, text="HSV Ranges", padx=3, pady=3)
        hsv_frame.pack(fill=X, pady=2)
        if color_name not in self.color_vars:
            self.color_vars[color_name] = {}
        self.color_vars[color_name]["hsv_scales"] = {}
        hsv_params = [
            ("H Min", 0, 179),("S Min", 0, 255),("V Min", 0, 255),
            ("H Max", 0, 179),("S Max", 0, 255),("V Max", 0, 255),
        ]
        for i, (text, min_val, max_val) in enumerate(hsv_params):
            r, c = i % 3, (i // 3) * 2
            Label(hsv_frame, text=text + ":").grid(row=r, column=c, sticky=W, padx=1, pady=1)
            scale = Scale(
                hsv_frame, from_=min_val, to=max_val, orient=HORIZONTAL, length=120, # Shorter scale
                command=lambda val, cn=color_name, p_idx=i: self.update_hsv_param_and_refresh(cn, p_idx, int(val)),
            )
            scale.grid(row=r, column=c + 1, sticky=EW, padx=1, pady=1)
            self.color_vars[color_name]["hsv_scales"][text.replace(" ", "_").lower()] = scale
            hsv_frame.grid_columnconfigure(c + 1, weight=1)

        morph_frame = LabelFrame(left_panel, text="Morphological Ops", padx=3, pady=3)
        morph_frame.pack(fill=X, pady=2)
        morph_ops = [
            ("morph_open_k", "Open K (odd,>=3):", 3, 21, 2), ("morph_open_iter", "Open Iter:", 0, 5, 1),
            ("morph_close_k", "Close K (odd,>=3):", 3, 21, 2), ("morph_close_iter", "Close Iter:", 0, 5, 1),
        ]
        if color_name in ["red", "blue"]:
            morph_ops.extend([
                ("morph_dilate_k", "Dilate K (odd,>=3):", 3, 21, 2), ("morph_dilate_iter", "Dilate Iter:", 0, 5, 1),
            ])
        for r_idx, (param_key, text, from_val, to_val, res) in enumerate(morph_ops):
            Label(morph_frame, text=text).grid(row=r_idx, column=0, sticky=W, padx=1, pady=1)
            scale = Scale(
                morph_frame, from_=from_val, to=to_val, orient=HORIZONTAL, resolution=res,
                command=lambda val, cn=color_name, pk=param_key: self.update_param_and_refresh(f"colors.{cn}.{pk}", int(val), cn),
            )
            scale.grid(row=r_idx, column=1, sticky=EW, padx=1, pady=1)
            self.color_vars[color_name][f"{param_key}_scale"] = scale
            morph_frame.grid_columnconfigure(1, weight=1)

        detection_params_frame = LabelFrame(left_panel, text="Detection Params", padx=3, pady=3)
        detection_params_frame.pack(fill=X, pady=2)
        
        # Entries for area
        det_entries_config = [("area_min", "Min Area:", int), ("area_max", "Max Area:", int)]
        if color_name in ["red", "blue"]:
            det_entries_config.extend([
                ("cluster_area_min", "Min Cluster Area:", int), ("cluster_area_max", "Max Cluster Area:", int),
            ])
        
        current_row = 0
        for param_key, text, type_func in det_entries_config:
            Label(detection_params_frame, text=text).grid(row=current_row, column=0, sticky=W, padx=1, pady=1)
            entry = Entry(detection_params_frame, width=7)
            entry.grid(row=current_row, column=1, sticky=W, padx=1, pady=1)
            entry.bind("<FocusOut>", lambda e, cn=color_name, pk=param_key, ent=entry, tf=type_func: self.update_entry_param_and_refresh(f"colors.{cn}.{pk}", ent.get(), cn, type_func=tf))
            entry.bind("<Return>", lambda e, cn=color_name, pk=param_key, ent=entry, tf=type_func: self.update_entry_param_and_refresh(f"colors.{cn}.{pk}", ent.get(), cn, type_func=tf))
            self.color_vars[color_name][f"{param_key}_entry"] = entry
            current_row +=1
            
        # Scales for circularity, solidity, aspect ratio
        det_scales_config = [("circularity", "Circularity (0-100%):", 100.0), ("solidity", "Solidity (0-100%):", 100.0)]
        if color_name in ["red", "blue"]:
            det_scales_config.extend([
                ("cluster_min_aspect_ratio", "Min Clust. Aspect (0-100%):", 100.0), # % of 1.0
                ("cluster_max_aspect_ratio", "Max Clust. Aspect (0-500%):", 100.0), # % of 1.0, e.g. 300 for 3.0
            ])
        
        for param_key, text, divisor in det_scales_config:
            to_val = 500 if "max_aspect" in param_key else 100
            Label(detection_params_frame, text=text).grid(row=current_row, column=0, sticky=W, padx=1, pady=1)
            scale = Scale(
                detection_params_frame, from_=0, to=to_val, orient=HORIZONTAL,
                command=lambda val, cn=color_name, pk=param_key, div=divisor: self.update_param_and_refresh(f"colors.{cn}.{pk}", float(val)/div, cn),
            )
            scale.grid(row=current_row, column=1, sticky=EW, padx=1, pady=1)
            self.color_vars[color_name][f"{param_key}_scale"] = scale
            current_row += 1
        detection_params_frame.grid_columnconfigure(1, weight=1)

        Label(right_panel, text=f"{color_name.capitalize()} Mask Preview:").pack(pady=(0, 3))
        mask_label = Label(right_panel, background="lightgrey", width=self.preview_size[0], height=self.preview_size[1])
        mask_label.pack()
        self.color_vars[f"{color_name}_mask_label"] = mask_label
        self.color_vars[f"{color_name}_mask_photo"] = None # To hold PhotoImage reference

    def update_param_and_refresh(self, param_path, value, color_to_refresh=None):
        keys = param_path.split(".")
        d = self.calib_params
        try:
            for key in keys[:-1]: d = d[key]
            if keys[-1] in ["morph_open_k", "morph_close_k", "morph_dilate_k", "blur_kernel"]:
                if value < 3: value = 3
                if value % 2 == 0: value = value + 1 if value > 0 else 3
            d[keys[-1]] = value
            if color_to_refresh: self.refresh_mask_display(color_to_refresh)
        except KeyError:
            print(f"Error: Invalid param path '{param_path}' during update.")
            traceback.print_exc()

    def update_param_and_refresh_all(self, param_path, value):
        keys = param_path.split(".")
        d = self.calib_params
        try:
            for key in keys[:-1]: d = d[key]
            if keys[-1] == "blur_kernel": # Specific handling for blur_kernel
                if value < 3: value = 3
                if value % 2 == 0: value = value + 1 if value > 0 else 3
            d[keys[-1]] = value
            self.refresh_all_mask_displays() # Global param change affects all masks
        except KeyError:
            print(f"Error: Invalid param path '{param_path}' during update all.")

    def update_entry_param_and_refresh(self, param_path, str_value, color_to_refresh, type_func=int):
        try:
            value = type_func(str_value)
            self.update_param_and_refresh(param_path, value, color_to_refresh)
        except ValueError:
            print(f"Invalid input for {param_path}: {str_value}. Not a valid {type_func.__name__}.")

    def update_hsv_param_and_refresh(self, color_name, param_index_flat, value):
        if not self.calib_params["colors"][color_name]["hsv_ranges"]:
            default_hsv_lower, default_hsv_upper = self.main_app.DEFAULT_DETECTION_PARAMS["colors"][color_name]["hsv_ranges"][0]
            self.calib_params["colors"][color_name]["hsv_ranges"] = [(default_hsv_lower.copy(), default_hsv_upper.copy())]
        
        current_lower, current_upper = self.calib_params["colors"][color_name]["hsv_ranges"][0]
        temp_lower, temp_upper = current_lower.copy(), current_upper.copy()
        
        param_map_to_channel_idx = {0:0, 1:1, 2:2, 3:0, 4:1, 5:2} # H,S,V Min, H,S,V Max
        is_lower = param_index_flat < 3
        channel_idx = param_map_to_channel_idx[param_index_flat]

        if is_lower: temp_lower[channel_idx] = value
        else: temp_upper[channel_idx] = value
        
        self.calib_params["colors"][color_name]["hsv_ranges"][0] = (temp_lower, temp_upper)
        self.refresh_mask_display(color_name)

    def load_params_to_ui(self):
        blur_k_val = self.calib_params.get("blur_kernel", 11)
        if blur_k_val % 2 == 0: blur_k_val = max(3, blur_k_val -1 if blur_k_val > 0 else 3)
        if "blur_kernel_scale" in self.color_vars and self.color_vars["blur_kernel_scale"].winfo_exists():
            self.color_vars["blur_kernel_scale"].set(blur_k_val)

        for color_name in ["white", "red", "blue"]:
            color_data = self.calib_params["colors"].get(color_name, {})
            default_color_data = self.main_app.DEFAULT_DETECTION_PARAMS["colors"][color_name]

            if color_data.get("hsv_ranges") and self.color_vars[color_name].get("hsv_scales"):
                if not all(k in self.color_vars[color_name]["hsv_scales"] for k in ["h_min","s_min","v_min","h_max","s_max","v_max"]):
                    continue
                if not color_data["hsv_ranges"]:
                    def_lower, def_upper = default_color_data["hsv_ranges"][0]
                    color_data["hsv_ranges"] = [(def_lower.copy(), def_upper.copy())]
                
                lower, upper = color_data["hsv_ranges"][0]
                self.color_vars[color_name]["hsv_scales"]["h_min"].set(lower[0])
                self.color_vars[color_name]["hsv_scales"]["s_min"].set(lower[1])
                self.color_vars[color_name]["hsv_scales"]["v_min"].set(lower[2])
                self.color_vars[color_name]["hsv_scales"]["h_max"].set(upper[0])
                self.color_vars[color_name]["hsv_scales"]["s_max"].set(upper[1])
                self.color_vars[color_name]["hsv_scales"]["v_max"].set(upper[2])

            morph_keys = ["morph_open_k","morph_open_iter","morph_close_k","morph_close_iter"]
            if color_name in ["red", "blue"]: morph_keys.extend(["morph_dilate_k","morph_dilate_iter"])
            for pk in morph_keys:
                if f"{pk}_scale" in self.color_vars[color_name] and self.color_vars[color_name][f"{pk}_scale"].winfo_exists():
                    val = color_data.get(pk, default_color_data.get(pk, 3 if "_k" in pk else 1))
                    if "_k" in pk and val % 2 == 0: val = max(3, val -1 if val > 0 else 3)
                    self.color_vars[color_name][f"{pk}_scale"].set(val)
            
            entry_pks = ["area_min", "area_max"]
            if color_name in ["red", "blue"]: entry_pks.extend(["cluster_area_min", "cluster_area_max"])
            for pk_entry in entry_pks:
                if f"{pk_entry}_entry" in self.color_vars[color_name] and self.color_vars[color_name][f"{pk_entry}_entry"].winfo_exists():
                    self.color_vars[color_name][f"{pk_entry}_entry"].delete(0, END)
                    def_val = default_color_data.get(pk_entry, 100 if "min" in pk_entry else (700 if "cluster" not in pk_entry else 5000))
                    self.color_vars[color_name][f"{pk_entry}_entry"].insert(0, str(color_data.get(pk_entry, def_val)))

            scale_pks_map = {"circularity":0.7, "solidity":0.7}
            if color_name in ["red","blue"]:
                scale_pks_map["cluster_min_aspect_ratio"] = default_color_data.get("cluster_min_aspect_ratio", 0.2)
                scale_pks_map["cluster_max_aspect_ratio"] = default_color_data.get("cluster_max_aspect_ratio", 3.0)
            for pk_scale, def_val_shape in scale_pks_map.items():
                 if f"{pk_scale}_scale" in self.color_vars[color_name] and self.color_vars[color_name][f"{pk_scale}_scale"].winfo_exists():
                    current_val = color_data.get(pk_scale, def_val_shape)
                    self.color_vars[color_name][f"{pk_scale}_scale"].set(int(current_val * 100))

    def generate_mask_for_color(self, color_name, image_to_process):
        if image_to_process is None or image_to_process.size == 0:
            return np.zeros((self.preview_size[1], self.preview_size[0]), dtype=np.uint8)

        params = self.calib_params["colors"].get(color_name, {})
        hsv_ranges = params.get("hsv_ranges", [])
        if not hsv_ranges:
            default_ranges = self.main_app.DEFAULT_DETECTION_PARAMS["colors"][color_name].get("hsv_ranges", [])
            hsv_ranges = copy.deepcopy(default_ranges) if default_ranges else []
            if not hsv_ranges: return np.zeros((self.preview_size[1], self.preview_size[0]), dtype=np.uint8)

        blur_k = self.calib_params.get("blur_kernel", 11)
        if blur_k < 3: blur_k = 3
        if blur_k % 2 == 0: blur_k = max(3, blur_k -1 if blur_k > 0 else 3)

        blurred_img = cv2.GaussianBlur(image_to_process, (blur_k, blur_k), 0)
        hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
        combined_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)

        for lower_hsv_np, upper_hsv_np in hsv_ranges:
            temp_lower, temp_upper = np.array(lower_hsv_np,dtype=np.uint8), np.array(upper_hsv_np,dtype=np.uint8)
            for i in range(1,3): # S,V
                if temp_upper[i] < temp_lower[i]: temp_upper[i] = temp_lower[i]
            if not (color_name == "red" and temp_lower[0] > temp_upper[0]):
                 if temp_upper[0] < temp_lower[0]: temp_upper[0] = temp_lower[0]
            
            seg_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
            if color_name == "red" and temp_lower[0] > temp_upper[0]:
                m1 = cv2.inRange(hsv_img, np.array([0, temp_lower[1],temp_lower[2]]), temp_upper)
                m2 = cv2.inRange(hsv_img, temp_lower, np.array([179, temp_upper[1],temp_upper[2]]))
                seg_mask = cv2.bitwise_or(m1, m2)
            else:
                seg_mask = cv2.inRange(hsv_img, temp_lower, temp_upper)
            combined_mask = cv2.bitwise_or(combined_mask, seg_mask)

        def_params = self.main_app.DEFAULT_DETECTION_PARAMS["colors"][color_name]
        op_k = params.get("morph_open_k", def_params.get("morph_open_k",5)); op_k=max(3,op_k+(1 if op_k%2==0 else 0))
        op_i = params.get("morph_open_iter", def_params.get("morph_open_iter",1))
        cl_k = params.get("morph_close_k", def_params.get("morph_close_k",5)); cl_k=max(3,cl_k+(1 if cl_k%2==0 else 0))
        cl_i = params.get("morph_close_iter", def_params.get("morph_close_iter", 2 if color_name=="white" else 1))
        
        morphed = combined_mask
        if op_i > 0: morphed=cv2.morphologyEx(morphed,cv2.MORPH_OPEN,np.ones((op_k,op_k),np.uint8),iterations=op_i)
        if color_name in ["red", "blue"]:
            di_k = params.get("morph_dilate_k", def_params.get("morph_dilate_k",5)); di_k=max(3,di_k+(1 if di_k%2==0 else 0))
            di_i = params.get("morph_dilate_iter", def_params.get("morph_dilate_iter",1))
            if di_i > 0: morphed=cv2.dilate(morphed,np.ones((di_k,di_k),np.uint8),iterations=di_i)
        if cl_i > 0: morphed=cv2.morphologyEx(morphed,cv2.MORPH_CLOSE,np.ones((cl_k,cl_k),np.uint8),iterations=cl_i)
        return morphed

    def refresh_mask_display(self, color_name):
        if self.img_to_process is None and self.main_app.img is not None: self.img_to_process = self.main_app.img.copy()
        elif self.main_app.img is None and self.img_to_process is not None: self.img_to_process = None

        if self.img_to_process is None or self.img_to_process.size == 0:
            mask_pil = Image.fromarray(np.zeros((self.preview_size[1], self.preview_size[0]), dtype=np.uint8), mode='L')
        else:
            mask = self.generate_mask_for_color(color_name, self.img_to_process)
            mask_resized = cv2.resize(mask, self.preview_size, interpolation=cv2.INTER_NEAREST)
            mask_pil = Image.fromarray(mask_resized, mode='L')
        
        try:
            photo_img = ImageTk.PhotoImage(image=mask_pil)
            self.color_vars[f"{color_name}_mask_photo"] = photo_img
            if f"{color_name}_mask_label" in self.color_vars and self.color_vars[f"{color_name}_mask_label"].winfo_exists():
                self.color_vars[f"{color_name}_mask_label"].config(image=photo_img)
        except Exception: pass # Avoid error if window closing

    def refresh_all_mask_displays(self):
        if self.main_app.img is not None:
            if self.img_to_process is None or self.img_to_process.shape!=self.main_app.img.shape or \
               (self.img_to_process.size>0 and self.main_app.img.size>0 and np.any(self.img_to_process!=self.main_app.img)):
                self.img_to_process = self.main_app.img.copy()
        elif self.img_to_process is not None: self.img_to_process = None
        for color_name in ["white", "red", "blue"]: self.refresh_mask_display(color_name)

    def refresh_all_mask_displays_from_button(self):
        if self.main_app.img is None:
            messagebox.showwarning("Refresh", "No image loaded in main app.")
            self.img_to_process = None
        else: self.img_to_process = self.main_app.img.copy()
        self.refresh_all_mask_displays()

    def apply_changes(self):
        self.main_app.active_detection_params = copy.deepcopy(self.calib_params)
        print("Info: Calibration params applied to main app.")
        if self.main_app.img is not None: self.main_app.run_full_detection_cycle(show_results_window=False)
        self.main_app.update_main_ui_detection_params_display()

    def reset_current_tab_to_defaults(self):
        try:
            # Correctly get color name from tab index
            current_tab_widget_name = self.notebook.select()
            current_tab_text = self.notebook.tab(current_tab_widget_name, "text").lower()
            color_name = current_tab_text # e.g. "white", "red", "blue"

            if color_name and color_name in self.main_app.DEFAULT_DETECTION_PARAMS["colors"]:
                if color_name == "white": # Global blur reset with white tab
                    self.calib_params["blur_kernel"] = copy.deepcopy(self.main_app.DEFAULT_DETECTION_PARAMS.get("blur_kernel", 11))
                self.calib_params["colors"][color_name] = copy.deepcopy(self.main_app.DEFAULT_DETECTION_PARAMS["colors"][color_name])
                self.load_params_to_ui() # Reload all UI for all tabs from updated calib_params
                self.refresh_mask_display(color_name) # Refresh current tab's mask
                if color_name == "white": self.refresh_all_mask_displays() # If global param changed
                print(f"Info: Params for {color_name} reset in calibration.")
        except Exception as e: 
            print(f"Error resetting tab: {e}")
            traceback.print_exc()

    def on_closing(self):
        self.main_app.calibration_window_open = False
        self.main_app.calibration_window = None
        self.destroy()

# --- Main Application Class ---
class FieldMeasureApp:
    PREVIEW_SCALE = 1.0
    LOUPE_SCALE = 2.0
    LOUPE_DIM = 160 # Slightly smaller loupe
    LOUPE_BORDER = 1
    LOUPE_BORDER_COLOR = (0, 255, 0); TARGET_COLOR = (0,0,255); TARGET_RADIUS = 2
    FIELD_W_CM = 400.0; FIELD_H_CM = 598.0
    PLC_IP = "192.168.0.200"; PLC_PORT = 2001; PLC_RECONNECT_INTERVAL = 5000

    DEFAULT_DETECTION_PARAMS = {
        "blur_kernel": 11,
        "colors": {
            "white": {"hsv_ranges":[(np.array([0,0,170]),np.array([180,65,255]))], "circularity":0.65,"solidity":0.75,
                      "area_min":100,"area_max":1500, "morph_open_k":5,"morph_open_iter":1,"morph_close_k":5,
                      "morph_close_iter":2, "primary_min_radius":6,"primary_circularity":0.65},
            "red": {"hsv_ranges":[(np.array([0,70,70]),np.array([10,255,255])),(np.array([170,70,70]),np.array([179,255,255]))],
                    "circularity":0.7,"solidity":0.65, "area_min":100,"area_max":700, "morph_open_k":5,"morph_open_iter":1,
                    "morph_dilate_k":5,"morph_dilate_iter":1,"morph_close_k":5,"morph_close_iter":1,
                    "cluster_area_min":600,"cluster_area_max":6000, "cluster_min_aspect_ratio":0.15,"cluster_max_aspect_ratio":6.0},
            "blue": {"hsv_ranges":[(np.array([100,70,70]),np.array([140,255,255]))], "circularity":0.65,"solidity":0.65,
                     "area_min":100,"area_max":700, "morph_open_k":5,"morph_open_iter":1,"morph_dilate_k":5,
                     "morph_dilate_iter":1,"morph_close_k":5,"morph_close_iter":1,
                     "cluster_area_min":600,"cluster_area_max":6000, "cluster_min_aspect_ratio":0.15,"cluster_max_aspect_ratio":6.0},
        },
        "detection_threshold_nms":0.01, "nms_overlap_threshold":0.3, # Adjusted NMS overlap
    }
    STATUS_CLICK_COLOR_INFO_MODE = False
    last_known_plc_distance=None; last_known_plc_angle=None; last_known_plc_swing_speed=None
    last_known_plc_release_speed=800; has_last_known_plc_data=False; sent_last_data_after_disappearance=False
    current_hsv_combined_mask_display = None
    MIN_16BIT_SIGNED = -32768; MAX_16BIT_SIGNED = 32767
    DEFAULT_COLOR_PICK_PATCH_SIZE = 5
    OBSTACLE_PROXIMITY_THRESHOLD_CM = 7.5 # Slightly reduced for finer check
    OPPONENT_NEAR_JACK_THRESHOLD_CM = 10.0 # For "very near" check, in cm
    TEAM_NONE=0; TEAM_RED=1; TEAM_BLUE=2 # Team identifiers

    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("Field Measurement Tool v6.1") # Incremented version

        self.img=None; self.preview=None; self.cap=None; self.camera_mode=False
        self.field_pts=[]; self.ball_pt1=None; self.ball_all=[]
        self.cursor_preview=(0,0); self.ball_detected=False; self.running=True
        self.camera_thread=None; self.canvas_photo=None; self.loupe_photo=None; self.combined_mask_photo=None
        self.pymc3e=None; self.plc_connected=False; self.plc_connecting=False; self.plc_attempt_reconnect=True
        self.default_detection_params=copy.deepcopy(self.DEFAULT_DETECTION_PARAMS)
        self.active_detection_params=copy.deepcopy(self.default_detection_params)
        self.calibration_window=None; self.calibration_window_open=False
        self.picked_color_info_list=[]

        self.target_x_cm_str=StringVar(value="153.0"); self.target_y_cm_str=StringVar(value="801.0")
        self.distance_display_str=StringVar(value="Dist: N/A"); self.angle_display_str=StringVar(value="Angle: N/A")
        self.ball_status_str=StringVar(value="Ball: Not found"); self.color_pick_patch_size_var=IntVar(value=self.DEFAULT_COLOR_PICK_PATCH_SIZE)
        wp = self.active_detection_params["colors"]["white"] # White params shorthand
        self.white_solidity_var=IntVar(value=int(wp["solidity"]*100)); self.white_circularity_var=IntVar(value=int(wp["circularity"]*100))
        self.white_min_radius_var=IntVar(value=wp.get("primary_min_radius",6))
        self.current_hsv_combined_mask_display=np.zeros((self.LOUPE_DIM,self.LOUPE_DIM,3),dtype=np.uint8)
        self.selecting_red_team_point=False; self.red_team_selected_point=None; self.red_team_xy_var=StringVar(value="X:-, Y:-")
        self.current_team = self.TEAM_NONE

        self.create_widgets() # This is the corrected method
        self._initialize_plc()
        self.update_main_ui_detection_params_display()
        self.root.after(self.PLC_RECONNECT_INTERVAL, self._check_and_reconnect_plc_job)

    def _initialize_plc(self):
        self.pymc3e = pymcprotocol.Type3E(); self.pymc3e.setaccessopt(commtype="binary")
        self.plc_attempt_reconnect = True; self._attempt_connect_plc()

    def _update_plc_gui_status(self, status_text, lamp_color):
        if hasattr(self,"plc_status_label_widget") and self.plc_status_label_widget.winfo_exists():
            self.plc_status_label_widget.config(text=f"PLC: {status_text}")
        if hasattr(self,"plc_lamp_canvas") and self.plc_lamp_canvas.winfo_exists():
            self.plc_lamp_canvas.itemconfig(self.plc_lamp_indicator, fill=lamp_color)

    def _update_plc_gui_status(self, status_text, lamp_color): # Existing method for context
        if hasattr(self, "plc_status_label_widget") and self.plc_status_label_widget.winfo_exists():
            self.plc_status_label_widget.config(text=f"PLC: {status_text}")
        if hasattr(self, "plc_lamp_canvas") and self.plc_lamp_canvas.winfo_exists():
            self.plc_lamp_canvas.itemconfig(self.plc_lamp_indicator, fill=lamp_color)

    # ADD THIS METHOD:
    def _generate_hsv_combined_mask_for_display(self):
        if self.img is None or self.img.size == 0:
            # Return a black image matching LOUPE_DIM for consistency
            return np.zeros((self.LOUPE_DIM, self.LOUPE_DIM, 3), dtype=np.uint8)

        try:
            hsv_image_orig = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
            # Create a colored mask for display
            colored_mask_preview_full_res = np.zeros_like(self.img) 

            colors_to_draw_ordered = [
                ("white", {"params_key": "white", "display_color": [0,255,0]}), # Green for white
                ("blue", {"params_key": "blue", "display_color": [255,0,0]}),   # Blue for blue
                ("red", {"params_key": "red", "display_color": [0,0,255]})     # Red for red
            ]

            for color_name_key, color_info in colors_to_draw_ordered:
                color_params = self.active_detection_params["colors"].get(color_info["params_key"])
                if not color_params or not color_params.get("hsv_ranges"):
                    continue
                
                hsv_ranges_list = color_params["hsv_ranges"]
                current_color_binary_mask_aggregated = np.zeros(hsv_image_orig.shape[:2], dtype=np.uint8)

                for lower_hsv_np, upper_hsv_np in hsv_ranges_list:
                    temp_lower = np.array(lower_hsv_np, dtype=np.uint8)
                    temp_upper = np.array(upper_hsv_np, dtype=np.uint8)

                    for i in range(1,3): # S, V
                        if temp_upper[i] < temp_lower[i]: temp_upper[i] = temp_lower[i]
                    if not (color_info["params_key"] == "red" and temp_lower[0] > temp_upper[0]): 
                        if temp_upper[0] < temp_lower[0]: temp_upper[0] = temp_lower[0]

                    individual_mask_segment = np.zeros(hsv_image_orig.shape[:2], dtype=np.uint8)
                    if color_info["params_key"] == "red" and temp_lower[0] > temp_upper[0]: 
                        mask1 = cv2.inRange(hsv_image_orig, np.array([0, temp_lower[1], temp_lower[2]]), temp_upper)
                        mask2 = cv2.inRange(hsv_image_orig, temp_lower, np.array([179, temp_upper[1], temp_upper[2]]))
                        individual_mask_segment = cv2.bitwise_or(mask1, mask2)
                    else:
                        individual_mask_segment = cv2.inRange(hsv_image_orig, temp_lower, temp_upper)
                    
                    current_color_binary_mask_aggregated = cv2.bitwise_or(current_color_binary_mask_aggregated, individual_mask_segment)
                
                colored_mask_preview_full_res[current_color_binary_mask_aggregated == 255] = color_info["display_color"]
            
            return colored_mask_preview_full_res 
        except Exception as e:
            print(f"Error in _generate_hsv_combined_mask_for_display: {e}")
            traceback.print_exc()
            return np.zeros((self.LOUPE_DIM, self.LOUPE_DIM, 3), dtype=np.uint8) # Fallback


    def _attempt_connect_plc(self):
        if self.plc_connecting or not self.pymc3e: return False
        self.plc_connecting = True; self._update_plc_gui_status("Connecting...", "orange")
        if hasattr(self.root,"update_idletasks") and self.root.winfo_exists(): self.root.update_idletasks()
        try:
            if self.plc_connected: self.pymc3e.close() # Close if already connected before reconnect
            self.pymc3e.connect(self.PLC_IP, self.PLC_PORT)
            self.plc_connected = True; self._update_plc_gui_status("Connected", "green")
            print(f"INFO: PLC Connected to {self.PLC_IP}:{self.PLC_PORT}")
            return True
        except Exception:
            self.plc_connected=False; self._update_plc_gui_status("Failed","red")
            return False
        finally: self.plc_connecting = False

    def _check_and_reconnect_plc_job(self):
        if not self.running: return
        if not self.plc_connected and self.plc_attempt_reconnect and not self.plc_connecting:
            self._attempt_connect_plc()
        if hasattr(self.root,"after") and self.root.winfo_exists():
            self.root.after(self.PLC_RECONNECT_INTERVAL, self._check_and_reconnect_plc_job)
            
    def create_widgets(self):
        # GUI Rearrangement for 1000x900 target
        self.image_frame = Frame(self.root) # Will take expand=True
        self.image_frame.pack(side=LEFT, padx=3, pady=3, fill=BOTH, expand=True)

        # Control panel with fixed width, height will be constrained by content
        self.control_frame = Frame(self.root, width=310) # Reduced width for controls
        self.control_frame.pack(side=RIGHT, padx=3, pady=3, fill=Y)
        self.control_frame.pack_propagate(False)
        
        # Canvas for image - corrected background color
        self.canvas = Canvas(self.image_frame, background="lightgrey") # Standard color
        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<Motion>", self.update_loupe_and_coords)
        self.canvas.bind("<Button-1>", self.handle_canvas_click)

        # Loupe and Coords Frame (below canvas)
        info_display_frame = Frame(self.image_frame)
        info_display_frame.pack(fill=X, pady=(2,0))
        self.combined_mask_label = Label(info_display_frame, bg="black", width=self.LOUPE_DIM, height=self.LOUPE_DIM)
        self.combined_mask_label.pack(side=LEFT, padx=1, pady=1, expand=True, fill=BOTH)
        # Initialize with blank image
        _blank_l_img = Image.new("RGB",(self.LOUPE_DIM,self.LOUPE_DIM),"black"); 
        self.combined_mask_photo=ImageTk.PhotoImage(_blank_l_img)
        self.combined_mask_label.config(image=self.combined_mask_photo)

        self.loupe_label = Label(info_display_frame, bg="lightgrey", width=self.LOUPE_DIM, height=self.LOUPE_DIM)
        self.loupe_label.pack(side=LEFT, padx=1, pady=1, expand=True, fill=BOTH)
        # Corrected blank image color for loupe_label
        _blank_r_img = Image.new("RGB",(self.LOUPE_DIM,self.LOUPE_DIM),"lightgrey");  # Use "lightgrey" or an RGB tuple
        self.loupe_photo=ImageTk.PhotoImage(_blank_r_img)
        self.loupe_label.config(image=self.loupe_photo)
        
        self.coord_label = Label(self.image_frame, text="Cursor: X: -, Y: -", font=("Arial", 8))
        self.coord_label.pack(pady=1, side=BOTTOM, fill=X)

        # --- Compact Control Panel Widgets ---
        sfont = ("Arial", 8); sfont_bold = ("Arial", 8, "bold"); sfont_small = ("Arial", 7)

        f_input = LabelFrame(self.control_frame,text="Input",font=sfont_bold); f_input.pack(fill=X,padx=2,pady=1)
        Button(f_input,text="Open Image",command=self.open_image_file,font=sfont).pack(fill=X,pady=1,ipady=0)
        Button(f_input,text="Open Camera",command=self.open_camera,font=sfont).pack(fill=X,pady=1,ipady=0)
        self.capture_btn=Button(f_input,text="Capture Frame",command=self.capture_frame,state=DISABLED,font=sfont)
        self.capture_btn.pack(fill=X,pady=1,ipady=0)

        f_red = LabelFrame(self.control_frame,text="Red Team Cmd",font=sfont_bold); f_red.pack(fill=X,padx=2,pady=1)
        Button(f_red,text="Select Red Pt",command=self.start_red_team_select_mode,bg="#FFCCCC",font=sfont).pack(fill=X,pady=1,ipady=0)
        self.red_team_xy_label=Label(f_red,textvariable=self.red_team_xy_var,font=sfont_bold,bg="white"); self.red_team_xy_label.pack(fill=X,pady=1)
        Button(f_red,text="Send Red Cmd",command=self.red_team,bg="red",fg="white",font=sfont).pack(fill=X,pady=1,ipady=0)

        f_corners = LabelFrame(self.control_frame,text="Field Corners",font=sfont_bold); f_corners.pack(fill=X,padx=2,pady=1)
        Label(f_corners,text="Order: TL,BL,TR,BR",font=("Arial",7,"italic")).pack(fill=X)
        self.corner_listbox=Listbox(f_corners,height=4,font=sfont_small); self.corner_listbox.pack(fill=X,pady=1)
        cf_btns=Frame(f_corners); cf_btns.pack(fill=X)
        Button(cf_btns,text="Del Last",command=self.remove_last_point,font=sfont).pack(side=LEFT,expand=True,fill=X,padx=1,ipady=0)
        Button(cf_btns,text="Clear All",command=self.clear_points,font=sfont).pack(side=LEFT,expand=True,fill=X,padx=1,ipady=0)

        f_detect = LabelFrame(self.control_frame,text="Detection",font=sfont_bold); f_detect.pack(fill=X,padx=2,pady=1)
        Button(f_detect,text="Detect Balls",command=lambda:self.run_full_detection_cycle(False),font=sfont).pack(fill=X,pady=1,ipady=0)
        self.ball_status_label=Label(f_detect,textvariable=self.ball_status_str,font=sfont); self.ball_status_label.pack(fill=X,pady=0)
        
        f_pick=Frame(f_detect); f_pick.pack(fill=X)
        Button(f_pick,text="Pick W",command=lambda:self.initiate_hsv_color_pick_for_params("white"),bg="#E0E0FF",font=sfont_small).pack(side=LEFT,expand=True,fill=X,padx=1,ipady=0)
        Button(f_pick,text="Pick R",command=lambda:self.initiate_hsv_color_pick_for_params("red"),bg="#FFE0E0",font=sfont_small).pack(side=LEFT,expand=True,fill=X,padx=1,ipady=0)
        Button(f_pick,text="Pick B",command=lambda:self.initiate_hsv_color_pick_for_params("blue"),bg="#E0FFE0",font=sfont_small).pack(side=LEFT,expand=True,fill=X,padx=1,ipady=0)
        
        Button(f_detect,text="Pick Color (Info)",command=self.toggle_info_color_pick_mode,font=sfont).pack(fill=X,pady=1,ipady=0)
        Button(f_detect,text="Advanced Params",command=self.open_calibration_window,bg="lightblue",font=sfont_bold).pack(fill=X,pady=1,ipady=0)
        
        f_patch=Frame(f_detect); f_patch.pack(fill=X,pady=(1,0))
        Label(f_patch,text="Pick Patch(px):",font=sfont_small).pack(side=LEFT,padx=(0,2))
        self.color_pick_patch_scale_main_ui=Scale(f_patch,from_=3,to=21,orient=HORIZONTAL,resolution=2,variable=self.color_pick_patch_size_var,length=70,font=sfont_small)
        self.color_pick_patch_scale_main_ui.pack(side=LEFT,fill=X,expand=True)

        f_wpri = LabelFrame(self.control_frame,text="Primary White (UI)",font=sfont_bold); f_wpri.pack(fill=X,padx=2,pady=1)
        Label(f_wpri,text="Solidity(%):",font=sfont_small).grid(row=0,column=0,sticky=W,pady=0)
        self.white_solidity_scale=Scale(f_wpri,from_=0,to=100,orient=HORIZONTAL,var=self.white_solidity_var,command=self.update_white_ball_detection_params_from_main_ui,length=60,font=sfont_small)
        self.white_solidity_scale.grid(row=0,column=1,sticky=EW,pady=0)
        Label(f_wpri,text="Circularity(%):",font=sfont_small).grid(row=1,column=0,sticky=W,pady=0)
        self.white_circularity_scale=Scale(f_wpri,from_=0,to=100,orient=HORIZONTAL,var=self.white_circularity_var,command=self.update_white_ball_detection_params_from_main_ui,length=60,font=sfont_small)
        self.white_circularity_scale.grid(row=1,column=1,sticky=EW,pady=0)
        Label(f_wpri,text="Min Radius(px):",font=sfont_small).grid(row=2,column=0,sticky=W,pady=0)
        self.white_min_radius_scale=Scale(f_wpri,from_=1,to=50,orient=HORIZONTAL,var=self.white_min_radius_var,command=self.update_white_ball_detection_params_from_main_ui,length=60,font=sfont_small)
        self.white_min_radius_scale.grid(row=2,column=1,sticky=EW,pady=0)
        f_wpri.grid_columnconfigure(1,weight=1)

        f_target = LabelFrame(self.control_frame,text="Target & Measure",font=sfont_bold); f_target.pack(fill=X,padx=2,pady=1)
        Label(f_target,text="Target X (cm):",font=sfont_small).grid(row=0,column=0,sticky=W,padx=2)
        self.x_entry=Entry(f_target,width=7,textvariable=self.target_x_cm_str,font=sfont_small); self.x_entry.grid(row=0,column=1,sticky=W,padx=2)
        Label(f_target,text="Target Y (cm):",font=sfont_small).grid(row=1,column=0,sticky=W,padx=2)
        self.y_entry=Entry(f_target,width=7,textvariable=self.target_y_cm_str,font=sfont_small); self.y_entry.grid(row=1,column=1,sticky=W,padx=2)
        Button(f_target,text="Set Target",command=self.set_target_position_action,font=sfont).grid(row=2,column=0,columnspan=2,pady=1,sticky="ew",ipady=0)
        Button(f_target,text="Calc & Show Detail",command=lambda:self.run_full_detection_cycle(True),bg="lightgreen",font=sfont_bold).grid(row=3,column=0,columnspan=2,pady=1,sticky="ew",ipady=0)
        self.distance_display_label=Label(f_target,textvariable=self.distance_display_str,font=("Arial",9,"bold")); self.distance_display_label.grid(row=4,column=0,columnspan=2,pady=0,sticky="w")
        self.angle_display_label=Label(f_target,textvariable=self.angle_display_str,font=("Arial",9,"bold")); self.angle_display_label.grid(row=5,column=0,columnspan=2,pady=0,sticky="w")
        f_target.grid_columnconfigure(1,weight=1)
        
        # PLC Status Frame - ensure it's at the bottom and visible
        f_plc = LabelFrame(self.control_frame, text="PLC Status", font=sfont_bold)
        f_plc.pack(fill=X, padx=2, pady=(5,1), side=BOTTOM) # Pack at bottom of control_frame
        
        f_plc_inner = Frame(f_plc)
        f_plc_inner.pack(pady=1,fill=X,expand=True)
        self.plc_lamp_canvas = Canvas(f_plc_inner, width=18, height=18) # Smaller lamp
        self.plc_lamp_canvas.pack(side=LEFT, padx=(3,2))
        self.plc_lamp_indicator = self.plc_lamp_canvas.create_oval(2,2,16,16,fill="grey",outline="black")
        self.plc_status_label_widget = Label(f_plc_inner, text="PLC: Init...", font=sfont)
        self.plc_status_label_widget.pack(side=LEFT, expand=True, fill=X)
        self._update_plc_gui_status("Initializing...", "grey")

    def _get_current_team(self):
        return self.current_team

    def _is_opponent_near_jack(self, jack_cm, opponent_balls_cm_coords, threshold_cm):
        if not opponent_balls_cm_coords or jack_cm is None: return False
        jx, jy = jack_cm
        for ox, oy in opponent_balls_cm_coords:
            if math.sqrt((jx - ox)**2 + (jy - oy)**2) < threshold_cm:
                return True
        return False

    def _is_obstacle_on_path(self, jack_cm, target_cm, obstacle_balls_cm_coords, proximity_threshold_cm):
        if not obstacle_balls_cm_coords or jack_cm is None or target_cm is None: return False
        p1_x, p1_y = jack_cm; t_x, t_y = target_cm
        vec_p1_t_x, vec_p1_t_y = t_x - p1_x, t_y - p1_y
        len_sq_p1_t = vec_p1_t_x**2 + vec_p1_t_y**2
        if len_sq_p1_t == 0: return False

        for obs_x, obs_y in obstacle_balls_cm_coords:
            line_A, line_B, line_C = t_y - p1_y, p1_x - t_x, (t_x * p1_y) - (t_y * p1_x)
            denominator = math.sqrt(line_A**2 + line_B**2)
            if denominator == 0: continue
            distance_to_line = abs(line_A * obs_x + line_B * obs_y + line_C) / denominator
            if distance_to_line < proximity_threshold_cm:
                dot_product = (obs_x - p1_x) * vec_p1_t_x + (obs_y - p1_y) * vec_p1_t_y
                if 0 < dot_product < len_sq_p1_t:
                    return True
        return False

    def update_white_ball_detection_params_from_main_ui(self, event=None):
        if "white" in self.active_detection_params["colors"]:
            wp = self.active_detection_params["colors"]["white"]
            wp["solidity"]=float(self.white_solidity_var.get())/100.0
            wp["circularity"]=float(self.white_circularity_var.get())/100.0
            wp["primary_circularity"]=wp["circularity"] 
            wp["primary_min_radius"]=self.white_min_radius_var.get()
            if self.calibration_window_open and self.calibration_window and self.calibration_window.winfo_exists():
                calib_wp = self.calibration_window.calib_params["colors"]["white"]
                calib_wp.update(wp) 
                self.calibration_window.load_params_to_ui()
                self.calibration_window.refresh_mask_display("white")
            if self.img is not None and not self.camera_mode: self.run_full_detection_cycle(False)
            elif self.img is not None and self.camera_mode:
                self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()

    def update_main_ui_detection_params_display(self):
        if hasattr(self, "white_solidity_var"):
            def_w = self.DEFAULT_DETECTION_PARAMS["colors"]["white"]
            act_w = self.active_detection_params["colors"].get("white", def_w)
            self.white_solidity_var.set(int(act_w.get("solidity",def_w["solidity"])*100))
            self.white_circularity_var.set(int(act_w.get("circularity",def_w["circularity"])*100))
            self.white_min_radius_var.set(act_w.get("primary_min_radius",def_w["primary_min_radius"]))

    def toggle_info_color_pick_mode(self):
        if self.img is None: messagebox.showwarning("Pick Color", "Open image/camera first."); return
        self.STATUS_CLICK_COLOR_INFO_MODE = not self.STATUS_CLICK_COLOR_INFO_MODE
        cursor_val = "crosshair" if self.STATUS_CLICK_COLOR_INFO_MODE else ""
        self.canvas.config(cursor=cursor_val)
        msg = f"Informational color pick mode {'ON' if self.STATUS_CLICK_COLOR_INFO_MODE else 'OFF'}."
        messagebox.showinfo("Pick Color Mode", msg)
        if self.STATUS_CLICK_COLOR_INFO_MODE and getattr(self,"picking_hsv_for_color",None) is not None:
            self.picking_hsv_for_color = None 

    def get_color_info_from_click(self, event):
        if self.img is None or self.preview is None: return
        if not (0<=event.x<self.preview.shape[1] and 0<=event.y<self.preview.shape[0]): return
        ix,iy=int(event.x/self.PREVIEW_SCALE),int(event.y/self.PREVIEW_SCALE)
        if not (0<=ix<self.img.shape[1] and 0<=iy<self.img.shape[0]): return
        bgr=self.img[iy,ix]; hsv=cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV)[0][0]
        print(f"--- Color Info ({ix},{iy}) BGR: {bgr}, HSV: {hsv} ---")
        self.picked_color_info_list.append({"bgr":bgr,"hsv":hsv,"coords":(ix,iy)})

    def set_target_position_action(self):
        try:
            x,y = float(self.target_x_cm_str.get()), float(self.target_y_cm_str.get())
            messagebox.showinfo("Target Set", f"Target: X:{x:.1f} cm, Y:{y:.1f} cm.")
            if self.img is not None and len(self.field_pts)==4: self.run_full_detection_cycle(False)
        except ValueError: messagebox.showerror("Invalid Input","Valid numbers for X,Y target.")

    def handle_canvas_click(self, event):
        if getattr(self,"selecting_red_team_point",False):
            X,Y=int(event.x/self.PREVIEW_SCALE),int(event.y/self.PREVIEW_SCALE)
            self.red_team_selected_point=(X,Y); self.red_team_xy_var.set(f"X:{X}, Y:{Y}")
            self.selecting_red_team_point=False; self.canvas.config(cursor="")
            messagebox.showinfo("Red Team", f"Selected Red Team point: ({X},{Y})")
            return
        if self.STATUS_CLICK_COLOR_INFO_MODE: self.get_color_info_from_click(event)
        elif getattr(self,"picking_hsv_for_color",None) is not None: self.process_hsv_color_pick(event)
        else: self.add_point(event)

    def open_image_file(self):
        self.stop_camera_if_running()
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[("Image files","*.jpg *.jpeg *.png *.bmp *.tiff"),("All files","*.*")]
        )
        if file_path:
            try:
                self.img = cv2.imread(file_path)
                if self.img is None: raise ValueError(f"Could not read image: {file_path}")
                self.reset_state_for_new_image_or_camera()
                self._dynamic_scale_and_set_preview()
                self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
                if self.calibration_window_open and self.calibration_window and self.calibration_window.winfo_exists():
                    self.calibration_window.refresh_all_mask_displays()
                self.update_main_canvas_display() 
            except Exception as e:
                messagebox.showerror("Image Error", f"Error opening image: {str(e)}")
                traceback.print_exc()
                self.img=None; self.preview=None
                self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
                self.update_main_canvas_display()

    def _dynamic_scale_and_set_preview(self):
        if self.img is None: self.preview=None; self.PREVIEW_SCALE=1.0; return
        if not hasattr(self.root,"winfo_exists") or not self.root.winfo_exists(): return
        self.root.update_idletasks()
        
        avail_w=self.image_frame.winfo_width(); avail_h=self.image_frame.winfo_height()
        info_h_est=self.LOUPE_DIM+10; coord_h_est=20 
        
        target_h=avail_h-info_h_est-coord_h_est-5 
        target_w=avail_w-5
        if target_w<=1: target_w=self.canvas.winfo_width() 
        if target_h<=1: target_h=self.canvas.winfo_height()
        if target_w<=1: target_w=600 
        if target_h<=1: target_h=400 

        img_h,img_w = self.img.shape[:2]
        if img_w==0 or img_h==0: self.preview=self.img.copy() if self.img is not None else None; self.PREVIEW_SCALE=1.0; return

        scale_w = target_w/img_w; scale_h = target_h/img_h
        self.PREVIEW_SCALE = min(scale_w,scale_h,1.0); self.PREVIEW_SCALE=max(0.1,self.PREVIEW_SCALE)
        
        prev_w=max(1,int(img_w*self.PREVIEW_SCALE)); prev_h=max(1,int(img_h*self.PREVIEW_SCALE))
        if prev_w>0 and prev_h>0: self.preview=cv2.resize(self.img,(prev_w,prev_h),interpolation=cv2.INTER_AREA)
        else: self.preview = None

    def open_camera(self):
        self.stop_camera_if_running()
        try:
            indices_to_try=[0,1,2,-1]; self.cap=None
            for i in indices_to_try:
                backend = cv2.CAP_DSHOW if sys.platform=="win32" else i
                temp_cap = cv2.VideoCapture(backend)
                if not temp_cap.isOpened() and backend!=i: temp_cap.release(); temp_cap=cv2.VideoCapture(i)
                if temp_cap.isOpened(): self.cap=temp_cap; print(f"INFO: Cam @ index {i} (backend:{backend}) opened."); break
                else: temp_cap.release()
            if not self.cap or not self.cap.isOpened(): raise ValueError("Could not open any camera.")

            res_set=False
            for w,h in [(2560,1440)]: 
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,w); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,h)
                time.sleep(0.2)
                aw,ah=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if abs(aw-w)<10 and abs(ah-h)<10: print(f"INFO: Cam res set to {aw}x{ah}"); res_set=True; break
            if not res_set: print(f"WARN: Using default cam res: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

            self.camera_mode=True; self.reset_state_for_new_image_or_camera()
            ret,frame=self.cap.read()
            if ret and frame is not None:
                self.img=frame.copy()
                if self.img is None or self.img.size==0: raise ValueError("Invalid initial cam frame.")
                self.running=True; self._dynamic_scale_and_set_preview()
                self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
                self.camera_thread=threading.Thread(target=self.update_camera_feed,daemon=True); self.camera_thread.start()
                if hasattr(self,"capture_btn"): self.capture_btn.config(state=NORMAL)
            else: raise ValueError("Could not grab initial frame from cam.")
        except Exception as e:
            print(f"ERR opening cam: {e}"); traceback.print_exc()
            if self.cap: self.cap.release(); self.cap=None
            self.running=False; self.camera_mode=False
            if hasattr(self,"capture_btn"): self.capture_btn.config(state=DISABLED)
            messagebox.showerror("Camera Error",f"Could not open/config cam: {e}")
            self.img=None;self.preview=None; self.current_hsv_combined_mask_display=self._generate_hsv_combined_mask_for_display()
            self.update_main_canvas_display()

    # (Place these methods inside your FieldMeasureApp class, replacing any existing versions)

    def update_camera_feed(self):
        last_error_report_time = 0
        error_report_interval = 5  # Seconds
        detection_interval = 0.05  # Run detection roughly every 50ms (20 FPS max for detection)
        last_detection_time = time.time()

        try:
            while self.running:
                if not (self.cap and self.cap.isOpened() and self.camera_mode):
                    break  # Exit if camera closed or not in camera mode
                
                ret, frame = self.cap.read()
                if not self.running:  # Check running flag again after read
                    break

                if ret and frame is not None:
                    self.img = frame.copy()  # Update main image buffer
                    if self.img is None or self.img.size == 0:
                        time.sleep(0.01)
                        continue  # Skip if frame is invalid

                    if not hasattr(self.root, "winfo_exists") or not self.root.winfo_exists():
                        break  # Root window closed, stop thread

                    self._dynamic_scale_and_set_preview()  # Update preview for display

                    # --- PLC Read Logic & Team Determination ---
                    m300_value = 0  # Default to TEAM_NONE or blue if 0
                    try:
                        if self.plc_connected and self.pymc3e:
                            read_values = self.pymc3e.batchread_wordunits(headdevice="M300", readsize=1) # Reading M300
                            if read_values:
                                m300_value = read_values[0]

                            # Determine team based on M300
                            if m300_value == 3: # Assuming 3 means Red Team
                                self.current_team = self.TEAM_RED
                            else: # Default to blue or none if not explicitly Red
                                self.current_team = self.TEAM_BLUE # Example: M300 not 3 could mean blue
                    except Exception as plc_e:
                        # print(f"PLC Read Error for M300 in camera_feed: {plc_e}")
                        self.current_team = self.TEAM_NONE # Fallback on PLC error
                    
                    # --- Detection and Measurement Logic ---
                    current_time = time.time()
                    if (current_time - last_detection_time) > detection_interval:
                        last_detection_time = current_time
                        
                        try:
                            self.detect_balls_in_frame() # This now handles cluster detection
                        except Exception as e_detect:
                            print(f"ERROR in detect_balls_in_frame during camera feed: {e_detect}")
                            traceback.print_exc() # Print stack trace for debugging
                            self.ball_all = [] # Clear detections to avoid using stale/bad data
                            self.ball_detected = False
                            self.ball_pt1 = None
                            # Optionally update ball status string here if needed
                            if hasattr(self.root, "after") and self.root.winfo_exists():
                                self.root.after(0, self.ball_status_str.set, "Ball: Detect Error")


                        if len(self.field_pts) == 4:  # Ensure field is defined
                            if self.ball_detected and self.ball_pt1 is not None:  # Jack ball found
                                measurement_data = self.process_measurements_for_realtime()
                                if measurement_data:
                                    final_dist_cm = measurement_data["distance_cm"]
                                    final_angle_deg = measurement_data["angle_degrees"]
                                    final_swing_speed = (final_dist_cm * 24.096) + 5900  # Default calculation
                                    final_release_speed = 800  # Default
                                    
                                    m190_value_to_set = 0 # Default M190 to OFF

                                    H_matrix_local = measurement_data.get("H_matrix")
                                    jack_cm_coords = (measurement_data["x1_cm"], measurement_data["y1_cm"])
                                    target_cm_coords = (measurement_data["x2_cm"], measurement_data["y2_cm"])
                                    
                                    opponent_balls_cm_in_roi = []
                                    if H_matrix_local is not None:
                                        for ball_info in self.ball_all: # self.ball_all is updated by detect_balls_in_frame
                                            is_opponent = (self.current_team == self.TEAM_RED and ball_info["color_name"] == "blue") or \
                                                          (self.current_team == self.TEAM_BLUE and ball_info["color_name"] == "red")
                                            if is_opponent:
                                                ball_pixel_center = np.float32([ball_info["center"]]).reshape(-1,1,2)
                                                ball_cm_transformed = cv2.perspectiveTransform(ball_pixel_center, H_matrix_local)
                                                if ball_cm_transformed is not None and ball_cm_transformed.size > 0:
                                                    opponent_balls_cm_in_roi.append(tuple(ball_cm_transformed[0,0]))
                                    
                                    obstacle_on_path = self._is_obstacle_on_path(jack_cm_coords, target_cm_coords, opponent_balls_cm_in_roi, self.OBSTACLE_PROXIMITY_THRESHOLD_CM)
                                    if obstacle_on_path:
                                        final_swing_speed = (final_dist_cm * 31.363) + 10864 - 1200
                                        final_release_speed = 500
                                        if self.current_team == self.TEAM_RED:
                                            final_angle_deg -= 2.0
                                        elif self.current_team == self.TEAM_BLUE:
                                            final_angle_deg += 2.0
                                    
                                    opponent_near_jack = self._is_opponent_near_jack(jack_cm_coords, opponent_balls_cm_in_roi, self.OPPONENT_NEAR_JACK_THRESHOLD_CM)
                                    if opponent_near_jack:
                                        final_swing_speed = (final_dist_cm * 555.55) + 666 
                                        final_release_speed = 777
                                        m190_value_to_set = 1 # Set M190 to ON 

                                    # Normalize angle
                                    final_angle_deg = (final_angle_deg + 360.0) % 360.0
                                    
                                    self.last_known_plc_distance = final_dist_cm
                                    self.last_known_plc_angle = final_angle_deg
                                    self.last_known_plc_swing_speed = final_swing_speed
                                    self.last_known_plc_release_speed = final_release_speed
                                    self.has_last_known_plc_data = True
                                    self.sent_last_data_after_disappearance = False

                                    if self.plc_connected and self.pymc3e:
                                        self.send_data_to_plc(self.last_known_plc_distance, self.last_known_plc_angle, 
                                                              self.last_known_plc_swing_speed, self.last_known_plc_release_speed)
                                        try:
                                            self.pymc3e.batchwrite_bitunits(headdevice="M190", values=[m190_value_to_set])
                                        except Exception as e_plc_m190:
                                            print(f"ERROR writing M190 to PLC (CameraFeed): {e_plc_m190}")
                                    
                                    display_data = measurement_data.copy() 
                                    display_data["distance_cm_final"] = self.last_known_plc_distance
                                    display_data["angle_degrees_final"] = self.last_known_plc_angle
                                    if hasattr(self.root, "after") and self.root.winfo_exists():
                                        self.root.after(0, self.update_measurement_display, display_data)
                                else: # measurement_data processing failed
                                    if hasattr(self.root, "after") and self.root.winfo_exists():
                                        self.root.after(0, self.update_measurement_display_default)
                            else: # Jack ball not detected
                                if hasattr(self.root, "after") and self.root.winfo_exists():
                                    self.root.after(0, self.update_measurement_display_default)
                                if self.has_last_known_plc_data and not self.sent_last_data_after_disappearance:
                                    if self.plc_connected and self.pymc3e: # Check pymc3e exists
                                        self.send_data_to_plc(self.last_known_plc_distance, self.last_known_plc_angle, 
                                                              self.last_known_plc_swing_speed, self.last_known_plc_release_speed)
                                        # Also turn M190 OFF if jack disappears
                                        try:
                                            self.pymc3e.batchwrite_bitunits(headdevice="M190", values=[0])
                                        except Exception as e_plc_m190_off:
                                            print(f"ERROR writing M190 OFF to PLC (Jack Disappeared): {e_plc_m190_off}")
                                    self.sent_last_data_after_disappearance = True
                        else: # Field not fully defined
                            if hasattr(self.root, "after") and self.root.winfo_exists():
                                self.root.after(0, self.update_measurement_display_default)
                    
                    # Schedule main canvas update (always, to show live video and detection overlays)
                    if hasattr(self.root, "after") and self.root.winfo_exists():
                        self.root.after(0, self.update_main_canvas_display_from_thread)
                else: # Frame not received
                    current_time_err = time.time()
                    if current_time_err - last_error_report_time > error_report_interval:
                        last_error_report_time = current_time_err
                        # print(f"Camera feed: No frame received at {time.strftime('%Y-%m-%d %H:%M:%S')}") # Optional debug
                    time.sleep(0.05) # Wait a bit longer if frames are not coming
                
                time.sleep(0.005) # Small delay to yield CPU, adjust as needed for performance
        
        except Exception as e_outer:
            if self.running and \
               ("application has been destroyed" not in str(e_outer).lower()) and \
               ("invalid command name" not in str(e_outer).lower()): # Avoid errors during shutdown
                print(f"CRITICAL ERROR in camera feed: {e_outer}")
                traceback.print_exc()
        finally:
            # Ensure cleanup or final UI state update is done on the main thread
            if hasattr(self.root, "winfo_exists") and self.root.winfo_exists() and hasattr(self.root, "after"):
                self.root.after(0, self.handle_camera_thread_exit)

    def run_full_detection_cycle(self, show_results_window=False):
        if self.img is None:
            if not show_results_window: # Avoid redundant message if detail window was intended
                messagebox.showwarning("Detection Error", "No image loaded.")
            self.ball_status_str.set("Ball: No Image")
            self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
            self.update_main_canvas_display()
            return

        # --- PLC Read for Team ---
        m300_value = 0
        try:
            if self.plc_connected and self.pymc3e:
                read_values = self.pymc3e.batchread_wordunits(headdevice="M300", readsize=1)
                if read_values:
                    m300_value = read_values[0]
                if m300_value == 3: # Assuming 3 means Red Team
                    self.current_team = self.TEAM_RED
                else: 
                    self.current_team = self.TEAM_BLUE 
        except Exception as plc_e:
            # print(f"PLC Read Error for M300 in full_cycle: {plc_e}")
            self.current_team = self.TEAM_NONE # Fallback

        try:
            self.detect_balls_in_frame()
        except Exception as e_detect_full:
            print(f"ERROR in detect_balls_in_frame (full cycle): {e_detect_full}")
            traceback.print_exc()
            self.ball_all = []
            self.ball_detected = False
            self.ball_pt1 = None
            self.ball_status_str.set("Ball: Detection Error")
            # Update canvas even on error to clear old detections
            self.update_main_canvas_display()
            return # Stop further processing if detection fails critically


        self.update_main_canvas_display() # Update visuals based on detection

        measurement_data_dict = None # To store results for potential detail window
        m190_value_to_set = 0 # Default M190 to OFF for this cycle

        if len(self.field_pts) == 4:
            if self.ball_detected and self.ball_pt1 is not None: # Primary white ball found
                measurement_data_dict = self.process_measurements_for_realtime()
                if measurement_data_dict:
                    final_dist_cm = measurement_data_dict["distance_cm"]
                    final_angle_deg = measurement_data_dict["angle_degrees"]
                    final_swing_speed = (final_dist_cm * 24.096) + 5900 # Default
                    final_release_speed = 800 # Default

                    H_matrix_local = measurement_data_dict.get("H_matrix")
                    jack_cm_coords = (measurement_data_dict["x1_cm"], measurement_data_dict["y1_cm"])
                    target_cm_coords = (measurement_data_dict["x2_cm"], measurement_data_dict["y2_cm"])
                    
                    opponent_balls_cm_in_roi = []
                    if H_matrix_local is not None:
                        for ball_info in self.ball_all:
                            is_opponent = (self.current_team == self.TEAM_RED and ball_info["color_name"] == "blue") or \
                                          (self.current_team == self.TEAM_BLUE and ball_info["color_name"] == "red")
                            if is_opponent:
                                ball_pixel_center = np.float32([ball_info["center"]]).reshape(-1,1,2)
                                ball_cm_transformed = cv2.perspectiveTransform(ball_pixel_center, H_matrix_local)
                                if ball_cm_transformed is not None and ball_cm_transformed.size > 0:
                                    opponent_balls_cm_in_roi.append(tuple(ball_cm_transformed[0,0]))
                    
                    obstacle_on_path = self._is_obstacle_on_path(jack_cm_coords, target_cm_coords, opponent_balls_cm_in_roi, self.OBSTACLE_PROXIMITY_THRESHOLD_CM)
                    if obstacle_on_path:
                        final_swing_speed = (final_dist_cm * 31.363) + 10864 - 1200
                        final_release_speed = 500
                        if self.current_team == self.TEAM_RED: final_angle_deg -= 2.0
                        elif self.current_team == self.TEAM_BLUE: final_angle_deg += 2.0
                    
                    opponent_near_jack = self._is_opponent_near_jack(jack_cm_coords, opponent_balls_cm_in_roi, self.OPPONENT_NEAR_JACK_THRESHOLD_CM)
                    if opponent_near_jack:
                        final_swing_speed = (final_dist_cm * 555.55) + 666
                        final_release_speed = 777
                        m190_value_to_set = 1 # Set M190 to ON
                    
                    final_angle_deg = (final_angle_deg + 360.0) % 360.0

                    self.last_known_plc_distance = final_dist_cm
                    self.last_known_plc_angle = final_angle_deg
                    self.last_known_plc_swing_speed = final_swing_speed
                    self.last_known_plc_release_speed = final_release_speed
                    self.has_last_known_plc_data = True
                    self.sent_last_data_after_disappearance = False

                    if self.plc_connected and self.pymc3e:
                        if not self.camera_mode: 
                            self.send_data_to_plc(self.last_known_plc_distance, self.last_known_plc_angle,
                                                  self.last_known_plc_swing_speed, self.last_known_plc_release_speed)
                        try:
                            self.pymc3e.batchwrite_bitunits(headdevice="M190", values=[m190_value_to_set])
                        except Exception as e_plc_m190_fc:
                            print(f"ERROR writing M190 to PLC (FullCycle): {e_plc_m190_fc}")
                    
                    display_data = measurement_data_dict.copy()
                    display_data["distance_cm_final"] = self.last_known_plc_distance
                    display_data["angle_degrees_final"] = self.last_known_plc_angle
                    self.update_measurement_display(display_data)
                else: # process_measurements_for_realtime failed
                    self.update_measurement_display_default()
                    # M190 should be OFF if measurements fail
                    if self.plc_connected and self.pymc3e:
                        try: self.pymc3e.batchwrite_bitunits(headdevice="M190", values=[0])
                        except Exception as e: print(f"Error M190 OFF (meas fail): {e}")

            else: # Primary white ball not detected or field not set
                self.update_measurement_display_default()
                if self.has_last_known_plc_data and not self.sent_last_data_after_disappearance:
                    if self.plc_connected and self.pymc3e:
                         self.send_data_to_plc(self.last_known_plc_distance, self.last_known_plc_angle,
                                               self.last_known_plc_swing_speed, self.last_known_plc_release_speed)
                         # M190 OFF if jack disappears and we send last known data
                         try: self.pymc3e.batchwrite_bitunits(headdevice="M190", values=[0])
                         except Exception as e: print(f"Error M190 OFF (jack gone): {e}")
                    self.sent_last_data_after_disappearance = True
                elif self.plc_connected and self.pymc3e: # Ensure M190 is off if no valid detection
                    try: self.pymc3e.batchwrite_bitunits(headdevice="M190", values=[0])
                    except Exception as e: print(f"Error M190 OFF (no detection): {e}")


        else: # Field corners not set
            self.update_measurement_display_default()
            if not self.camera_mode and not show_results_window: 
                 if hasattr(self, "ball_status_str"): self.ball_status_str.set("Ball: Define 4 field corners")
            # Ensure M190 is off if field not set
            if self.plc_connected and self.pymc3e:
                try: self.pymc3e.batchwrite_bitunits(headdevice="M190", values=[0])
                except Exception as e: print(f"Error M190 OFF (no field): {e}")

        
        if show_results_window:
            if len(self.field_pts) != 4:
                messagebox.showwarning("Results Error", "4 field corners must be selected for detailed results.")
                return
            if not self.ball_detected or self.ball_pt1 is None:
                messagebox.showwarning("Results Error", "Primary white ball not detected. Cannot show details.")
                return
            if measurement_data_dict is None: 
                measurement_data_dict = self.process_measurements_for_realtime() 
            
            if measurement_data_dict:
                self.show_detailed_results_window(measurement_data_dict)
            else:
                messagebox.showerror("Results Error","Could not process measurements for detail view.")

    def stop_camera_if_running(self):
        self.running = False
        if self.camera_thread and self.camera_thread.is_alive(): self.camera_thread.join(timeout=0.5)
        self.camera_thread = None
        if self.cap: self.cap.release(); self.cap = None
        self.camera_mode = False
        if hasattr(self,"capture_btn") and self.capture_btn.winfo_exists(): self.capture_btn.config(state=DISABLED)

    def capture_frame(self):
        if self.img is None: messagebox.showwarning("Capture","No image/camera active."); return
        self.run_full_detection_cycle(show_results_window=True)

    def update_main_canvas_display_from_thread(self):
        if (self.preview is not None or self.img is None) and hasattr(self.canvas,"winfo_exists") and self.canvas.winfo_exists():
            self.update_main_canvas_display()

    def handle_camera_thread_exit(self):
        if hasattr(self,"capture_btn") and self.capture_btn.winfo_exists(): self.capture_btn.config(state=DISABLED)
        print("Camera thread has exited.")

    def detect_balls_in_frame(self):
        if self.img is None or self.img.size == 0:
            self.ball_pt1, self.ball_all, self.ball_detected = None, [], False
            self.ball_status_str.set("Ball: No image")
            self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
            return

        if len(self.field_pts) != 4:
            self.ball_pt1, self.ball_all, self.ball_detected = None, [], False
            if hasattr(self,"ball_status_str"): self.ball_status_str.set("Ball: Define 4 corners")
            self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
            return

        self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
        img_for_detection = self.img.copy()

        field_mask_cv = np.zeros(self.img.shape[:2], dtype=np.uint8)
        try:
            pts_for_poly = np.array([self.field_pts[0],self.field_pts[1],self.field_pts[3],self.field_pts[2]], dtype=np.int32)
            cv2.fillPoly(field_mask_cv, [pts_for_poly], 255)
            img_for_detection = cv2.bitwise_and(img_for_detection,img_for_detection,mask=field_mask_cv)
        except Exception as e_mask:
            print(f"Error creating field mask: {e_mask}")

        self.ball_pt1 = None 
        self.ball_detected = False 
        all_detected_objects_for_nms = [] 

        blur_k = self.active_detection_params.get("blur_kernel", 11)
        blur_k = max(3, blur_k + (1 if blur_k % 2 == 0 else 0))
        try:
            blurred = cv2.GaussianBlur(img_for_detection, (blur_k, blur_k), 0)
            hsv_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        except Exception as e_blur:
            print(f"Error during blur/HSV conversion: {e_blur}")
            hsv_blurred = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2HSV) 

        best_hsv_white_ball_for_primary = None
        highest_primary_white_metric = -1.0
        
        wp = self.active_detection_params["colors"].get("white", self.DEFAULT_DETECTION_PARAMS["colors"]["white"])
        if wp:
            hsv_ranges_list = wp.get("hsv_ranges", [])
            if hsv_ranges_list:
                mask_w = np.zeros(hsv_blurred.shape[:2],dtype=np.uint8)
                for lr, ur in hsv_ranges_list:
                    l=np.array(lr,dtype=np.uint8); u=np.array(ur,dtype=np.uint8)
                    for ch_idx in range(3): 
                        if u[ch_idx] < l[ch_idx]: u[ch_idx] = l[ch_idx]
                    mask_w = cv2.bitwise_or(mask_w, cv2.inRange(hsv_blurred, l, u))

                op_k_w=max(3,wp.get("morph_open_k",5)+(1 if wp.get("morph_open_k",5)%2==0 else 0))
                op_i_w=wp.get("morph_open_iter",1)
                cl_k_w=max(3,wp.get("morph_close_k",5)+(1 if wp.get("morph_close_k",5)%2==0 else 0))
                cl_i_w=wp.get("morph_close_iter",2)
                
                if op_i_w>0: mask_w=cv2.morphologyEx(mask_w,cv2.MORPH_OPEN,np.ones((op_k_w,op_k_w),np.uint8),iterations=op_i_w)
                if cl_i_w>0: mask_w=cv2.morphologyEx(mask_w,cv2.MORPH_CLOSE,np.ones((cl_k_w,cl_k_w),np.uint8),iterations=cl_i_w)
                
                contours_w, _ = cv2.findContours(mask_w, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_a_w, max_a_w = wp.get("area_min",100), wp.get("area_max",1500)
                circ_t_w, sol_t_w = wp.get("circularity",0.65), wp.get("solidity",0.75)
                prim_circ_t, prim_rad_t = wp.get("primary_circularity",0.65), wp.get("primary_min_radius",6)

                for cnt in contours_w:
                    area = cv2.contourArea(cnt)
                    if not (min_a_w <= area <= max_a_w): continue
                    perimeter = cv2.arcLength(cnt,True);
                    if perimeter == 0: continue
                    circularity = 4 * np.pi * area / (perimeter**2)
                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    hull=cv2.convexHull(cnt); hull_a=cv2.contourArea(hull)
                    solidity = float(area)/hull_a if hull_a > 0 else 0

                    if circularity >= circ_t_w and solidity >= sol_t_w:
                        conf = circularity*0.5 + solidity*0.3 + min(1.0, area/max_a_w)*0.2
                        obj = {"bbox":(int(x-radius),int(y-radius),int(x+radius),int(y+radius)), "confidence":conf,
                               "center":(int(x),int(y)), "radius":int(radius), "color_name":"white", "shape_type":"circle",
                               "circularity":circularity, "solidity":solidity, "area":area, "is_primary_white": False}
                        all_detected_objects_for_nms.append(obj)
                        
                        if circularity >= prim_circ_t and radius >= prim_rad_t:
                            metric = circularity*1000 + radius 
                            if metric > highest_primary_white_metric:
                                highest_primary_white_metric = metric
                                best_hsv_white_ball_for_primary = obj

        primary_jack_center_px = None # Store (px, py) of jack center for ROI
        primary_jack_radius_px = 0

        if best_hsv_white_ball_for_primary:
            self.ball_detected = True
            best_hsv_white_ball_for_primary["is_primary_white"] = True 
            primary_jack_center_px = best_hsv_white_ball_for_primary["center"]
            primary_jack_radius_px = best_hsv_white_ball_for_primary["radius"]
            self.ball_pt1 = (primary_jack_center_px[0], primary_jack_center_px[1] + primary_jack_radius_px) 
            self.ball_status_str.set(f"W:Found (C:{best_hsv_white_ball_for_primary['circularity']:.2f} R:{primary_jack_radius_px})")
            
            ROI_RADIUS_FACTOR = 12.0 
            search_radius_sq_px = (primary_jack_radius_px * ROI_RADIUS_FACTOR)**2

            for color_name in ["red", "blue"]:
                cp = self.active_detection_params["colors"].get(color_name, self.DEFAULT_DETECTION_PARAMS["colors"][color_name])
                hsv_ranges_list_rb = cp.get("hsv_ranges",[])
                if not hsv_ranges_list_rb: continue

                mask_rb = np.zeros(hsv_blurred.shape[:2],dtype=np.uint8)
                for lr,ur in hsv_ranges_list_rb:
                    l,u=np.array(lr,dtype=np.uint8),np.array(ur,dtype=np.uint8)
                    for ch_idx in range(1,3): 
                        if u[ch_idx] < l[ch_idx]: u[ch_idx] = l[ch_idx]
                    if not(color_name=="red" and l[0]>u[0]): 
                        if u[0] < l[0]: u[0] = l[0]
                    
                    seg_mask_rb = np.zeros(hsv_blurred.shape[:2],dtype=np.uint8)
                    if color_name=="red" and l[0]>u[0]: 
                        m1=cv2.inRange(hsv_blurred,np.array([0,l[1],l[2]]),u)
                        m2=cv2.inRange(hsv_blurred,l,np.array([179,u[1],u[2]]))
                        seg_mask_rb = cv2.bitwise_or(m1,m2)
                    else: seg_mask_rb = cv2.inRange(hsv_blurred,l,u)
                    mask_rb = cv2.bitwise_or(mask_rb, seg_mask_rb)

                op_k_rb=max(3,cp.get("morph_open_k",5)+(1 if cp.get("morph_open_k",5)%2==0 else 0))
                op_i_rb=cp.get("morph_open_iter",1)
                di_k_rb=max(3,cp.get("morph_dilate_k",5)+(1 if cp.get("morph_dilate_k",5)%2==0 else 0))
                di_i_rb=cp.get("morph_dilate_iter",1)
                cl_k_rb=max(3,cp.get("morph_close_k",5)+(1 if cp.get("morph_close_k",5)%2==0 else 0))
                cl_i_rb=cp.get("morph_close_iter",1)

                if op_i_rb>0: mask_rb=cv2.morphologyEx(mask_rb,cv2.MORPH_OPEN,np.ones((op_k_rb,op_k_rb),np.uint8),iterations=op_i_rb)
                if di_i_rb>0: mask_rb=cv2.dilate(mask_rb,np.ones((di_k_rb,di_k_rb),np.uint8),iterations=di_i_rb)
                if cl_i_rb>0: mask_rb=cv2.morphologyEx(mask_rb,cv2.MORPH_CLOSE,np.ones((cl_k_rb,cl_k_rb),np.uint8),iterations=cl_i_rb)

                contours_rb, _ = cv2.findContours(mask_rb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                min_a_rb,max_a_rb=cp.get("area_min",100),cp.get("area_max",700)
                circ_t_rb,sol_t_rb=cp.get("circularity",0.6),cp.get("solidity",0.6)
                c_min_a, c_max_a = cp.get("cluster_area_min",600), cp.get("cluster_area_max",6000)
                c_min_ar, c_max_ar = cp.get("cluster_min_aspect_ratio",0.15), cp.get("cluster_max_aspect_ratio",6.0)

                for cnt_rb in contours_rb:
                    M=cv2.moments(cnt_rb)
                    if M["m00"]==0: continue
                    cx_rb,cy_rb=int(M["m10"]/M["m00"]),int(M["m01"]/M["m00"])
                    if (cx_rb-primary_jack_center_px[0])**2 + (cy_rb-primary_jack_center_px[1])**2 > search_radius_sq_px: continue 

                    area_rb = cv2.contourArea(cnt_rb)
                    
                    is_single_ball = False
                    if min_a_rb <= area_rb <= max_a_rb:
                        perimeter_rb=cv2.arcLength(cnt_rb,True)
                        if perimeter_rb>0:
                            circ_rb=4*np.pi*area_rb/(perimeter_rb**2)
                            (x_rb,y_rb),rad_rb=cv2.minEnclosingCircle(cnt_rb)
                            hull_rb=cv2.convexHull(cnt_rb); hull_a_rb=cv2.contourArea(hull_rb)
                            sol_rb = float(area_rb)/hull_a_rb if hull_a_rb>0 else 0
                            if circ_rb >= circ_t_rb and sol_rb >= sol_t_rb:
                                conf_rb = circ_rb*0.5 + sol_rb*0.3 + min(1.0, area_rb/max_a_rb)*0.2
                                all_detected_objects_for_nms.append({
                                    "bbox":(int(x_rb-rad_rb),int(y_rb-rad_rb),int(x_rb+rad_rb),int(y_rb+rad_rb)),
                                    "confidence":conf_rb, "center":(int(x_rb),int(y_rb)), "radius":int(rad_rb),
                                    "color_name":color_name, "shape_type":"circle", "area":area_rb
                                })
                                is_single_ball = True
                    
                    if not is_single_ball and c_min_a <= area_rb <= c_max_a: # Check for cluster
                        x_br, y_br, w_br, h_br = cv2.boundingRect(cnt_rb)
                        aspect_ratio_br = float(w_br)/h_br if h_br > 0 else 0.0 # Avoid div by zero
                        # Ensure w_br is also > 0 for the inverse aspect ratio if h_br is 0
                        inv_aspect_ratio_br = float(h_br)/w_br if w_br > 0 else 0.0
                        
                        is_valid_aspect = (c_min_ar <= aspect_ratio_br <= c_max_ar) or \
                                          (c_min_ar <= inv_aspect_ratio_br <= c_max_ar)
                        
                        if is_valid_aspect:
                            conf_cl = 0.4 + min(1.0, (area_rb - c_min_a) / (c_max_a - c_min_a + 1e-6)) * 0.6 # Confidence for cluster
                            all_detected_objects_for_nms.append({
                                "bbox":(x_br,y_br,x_br+w_br,y_br+h_br), "rect_coords": (x_br,y_br,w_br,h_br),
                                "confidence":conf_cl, "center":(cx_rb,cy_rb), 
                                "color_name":color_name, "shape_type":"rectangle", "area":area_rb
                            })
        elif not self.ball_detected: 
            self.ball_status_str.set("Ball: White not found")

        self.ball_all = []
        if all_detected_objects_for_nms:
            boxes_nms = []; confs_nms = []; orig_indices = []
            for i, obj in enumerate(all_detected_objects_for_nms):
                x1,y1,x2,y2 = obj["bbox"]; boxes_nms.append([x1,y1,x2-x1,y2-y1]); confs_nms.append(obj["confidence"]); orig_indices.append(i)
            
            if boxes_nms:
                try:
                    nms_idx = cv2.dnn.NMSBoxes(boxes_nms, np.array(confs_nms).astype(np.float32),
                                               self.active_detection_params.get("detection_threshold_nms",0.01),
                                               self.active_detection_params.get("nms_overlap_threshold",0.3))
                    if isinstance(nms_idx, np.ndarray):
                        if nms_idx.ndim > 1: nms_idx = nms_idx.flatten()
                        for idx_val in nms_idx: self.ball_all.append(all_detected_objects_for_nms[orig_indices[idx_val]])
                except Exception as e_nms: print(f"NMS Error: {e_nms}"); self.ball_all = all_detected_objects_for_nms

        if self.ball_detected and best_hsv_white_ball_for_primary:
            is_pri_in_list = any(b.get("is_primary_white",False) for b in self.ball_all)
            if not is_pri_in_list:
                for b_obj in self.ball_all: # Ensure no other white ball is accidentally marked primary
                    if b_obj["color_name"]=="white" and b_obj.get("is_primary_white"): b_obj["is_primary_white"]=False
                self.ball_all.append(best_hsv_white_ball_for_primary)
        
        if not self.ball_all and not self.ball_detected and not self.ball_status_str.get().startswith("Ball: Define 4"):
             self.ball_status_str.set("Ball: None found")

    def update_main_canvas_display(self):
        if self.preview is None:
            if hasattr(self.canvas,"winfo_exists") and self.canvas.winfo_exists():
                self.canvas.delete("all")
                cw,ch=self.canvas.winfo_width(),self.canvas.winfo_height()
                if cw<=1 or ch<=1: cw,ch=int(self.canvas.cget("width")),int(self.canvas.cget("height"))
                if cw>1 and ch>1: self.canvas.create_text(cw//2,ch//2,text="No image / Cam off",font=("Arial",14))
            if hasattr(self.combined_mask_label,"winfo_exists") and self.combined_mask_label.winfo_exists():
                mask_disp = cv2.resize(self.current_hsv_combined_mask_display, (self.LOUPE_DIM,self.LOUPE_DIM), interpolation=cv2.INTER_NEAREST)
                mask_rgb = cv2.cvtColor(mask_disp, cv2.COLOR_BGR2RGB)
                mask_pil = Image.fromarray(mask_rgb)
                self.combined_mask_photo = ImageTk.PhotoImage(image=mask_pil)
                self.combined_mask_label.config(image=self.combined_mask_photo)
            return

        draw_preview = self.preview.copy()
        for i,pt_orig in enumerate(self.field_pts):
            x_prev,y_prev=int(pt_orig[0]*self.PREVIEW_SCALE),int(pt_orig[1]*self.PREVIEW_SCALE)
            cv2.circle(draw_preview,(x_prev,y_prev),4,(0,255,0),-1) 
            cv2.putText(draw_preview,str(i+1),(x_prev+5,y_prev+5),cv2.FONT_HERSHEY_SIMPLEX,0.35,(255,255,255),1)

        ball_draw_colors = {"white":(230,230,230),"red":(0,0,255),"blue":(255,100,100),"default":(0,200,0)}
        for det_obj in self.ball_all:
            center_orig = det_obj["center"] 
            color_name = det_obj["color_name"]
            shape_type = det_obj.get("shape_type", "circle") 
            
            center_prev = (int(center_orig[0]*self.PREVIEW_SCALE), int(center_orig[1]*self.PREVIEW_SCALE))
            draw_clr = ball_draw_colors.get(color_name, ball_draw_colors["default"])

            if shape_type == "circle":
                radius_orig = det_obj.get("radius", 5) 
                radius_prev = int(max(2, radius_orig * self.PREVIEW_SCALE))
                cv2.circle(draw_preview, center_prev, radius_prev, draw_clr, 1) 
                label_txt = f"{color_name[0].upper()}"
                if det_obj.get("is_primary_white"): label_txt = "Jack"
                cv2.putText(draw_preview,label_txt,(center_prev[0]-radius_prev//2, center_prev[1]-radius_prev-2), cv2.FONT_HERSHEY_SIMPLEX,0.3,draw_clr,1)
            elif shape_type == "rectangle": # Drawing cluster as rectangle
                x_br,y_br,w_br,h_br = det_obj["rect_coords"]
                x1_p,y1_p = int(x_br*self.PREVIEW_SCALE), int(y_br*self.PREVIEW_SCALE)
                x2_p,y2_p = int((x_br+w_br)*self.PREVIEW_SCALE), int((y_br+h_br)*self.PREVIEW_SCALE)
                cv2.rectangle(draw_preview, (x1_p,y1_p), (x2_p,y2_p), draw_clr, 1) # Simple rectangle
                # Rounded rectangle for "soft edge" is more complex; Pillow draw might be better if needed.
                # For now, a simple rectangle.
                cv2.putText(draw_preview, f"{color_name[0].upper()}Cl", (x1_p,y1_p-2),cv2.FONT_HERSHEY_SIMPLEX,0.3,draw_clr,1)

        if self.ball_detected and self.ball_pt1: 
            p1_mx,p1_my = int(self.ball_pt1[0]*self.PREVIEW_SCALE), int(self.ball_pt1[1]*self.PREVIEW_SCALE)
            cv2.drawMarker(draw_preview,(p1_mx,p1_my),(255,255,0),markerType=cv2.MARKER_CROSS,markerSize=8,thickness=1)
            
        img_pil_canvas = Image.fromarray(cv2.cvtColor(draw_preview,cv2.COLOR_BGR2RGB))
        self.canvas_photo = ImageTk.PhotoImage(image=img_pil_canvas)
        if hasattr(self.canvas,"winfo_exists") and self.canvas.winfo_exists():
            self.canvas.create_image(0,0,anchor=NW,image=self.canvas_photo)

        if hasattr(self,"current_hsv_combined_mask_display") and self.current_hsv_combined_mask_display is not None:
            try:
                mask_disp = cv2.resize(self.current_hsv_combined_mask_display,(self.LOUPE_DIM,self.LOUPE_DIM),interpolation=cv2.INTER_NEAREST)
                mask_rgb = cv2.cvtColor(mask_disp,cv2.COLOR_BGR2RGB)
                mask_pil_tk = Image.fromarray(mask_rgb)
                self.combined_mask_photo = ImageTk.PhotoImage(image=mask_pil_tk)
                if hasattr(self.combined_mask_label,"winfo_exists") and self.combined_mask_label.winfo_exists():
                    self.combined_mask_label.config(image=self.combined_mask_photo)
            except Exception: pass 

    def add_point(self, event):
        if self.img is None or self.preview is None: messagebox.showwarning("Add Point","No image."); return
        if not (0<=event.x<self.preview.shape[1] and 0<=event.y<self.preview.shape[0]): return
        if len(self.field_pts) < 4:
            Xo,Yo=int(event.x/self.PREVIEW_SCALE),int(event.y/self.PREVIEW_SCALE)
            if not (0<=Xo<self.img.shape[1] and 0<=Yo<self.img.shape[0]): return
            self.field_pts.append((Xo,Yo))
            self.corner_listbox.insert(END,f"P{len(self.field_pts)}: ({Xo},{Yo})")
            if len(self.field_pts)==4: self.has_last_known_plc_data=False; self.run_full_detection_cycle(False)
            else: self.update_main_canvas_display()
        else: messagebox.showinfo("Add Point","4 corners already set. Clear to reselect.")

    def remove_last_point(self):
        if self.field_pts:
            self.field_pts.pop(); self.corner_listbox.delete(END)
            if len(self.field_pts)<4:
                self.ball_pt1,self.ball_all,self.ball_detected=None,[],False
                self.current_hsv_combined_mask_display=self._generate_hsv_combined_mask_for_display()
                self.ball_status_str.set("Ball: Define 4 corners")
                self.update_measurement_display_default(); self.has_last_known_plc_data=False
            self.update_main_canvas_display()
        else: messagebox.showinfo("Remove Point","No points to remove.")

    def clear_points(self):
        if not self.field_pts: messagebox.showinfo("Clear Points","No points to clear."); return
        if messagebox.askyesno("Confirm Clear","Clear all field corner points?"):
            self.field_pts=[]; self.corner_listbox.delete(0,END)
            self.ball_pt1,self.ball_all,self.ball_detected=None,[],False
            self.current_hsv_combined_mask_display=self._generate_hsv_combined_mask_for_display()
            self.ball_status_str.set("Ball: Define 4 corners")
            self.update_main_canvas_display(); self.update_measurement_display_default(); self.has_last_known_plc_data=False

    def _float_to_rounded_int_word_list(self, float_val, value_name="value"):
        try:
            if float_val is None or math.isnan(float_val) or math.isinf(float_val): return [0]
            rnd_int = int(round(float_val))
            return [max(self.MIN_16BIT_SIGNED, min(self.MAX_16BIT_SIGNED, rnd_int))]
        except: return [0]

    def send_data_to_plc(self, dist_val, angle_val, swing_val, release_val):
        if not self.plc_connected or not self.pymc3e:
            # print("PLC not connected, cannot send data.") # Optional debug
            return
        try:
            d_val = self._float_to_rounded_int_word_list(dist_val)[0]
            a_val = self._float_to_rounded_int_word_list(angle_val)[0]
            s_val = self._float_to_rounded_int_word_list(swing_val)[0]
            r_val = self._float_to_rounded_int_word_list(release_val)[0]
            
            # Corrected call to randomwrite with all required arguments
            self.pymc3e.randomwrite(
                word_devices=["D1", "D120", "D106", "D108"], 
                word_values=[d_val, a_val, s_val, r_val],
                dword_devices=[],  # Provide empty list for dword_devices
                dword_values=[]    # Provide empty list for dword_values
            )
            self._update_plc_gui_status("Data Sent", "green")
        except Exception as e_plc_tx:
            print(f"ERR: PLC write error: {e_plc_tx}")
            self.plc_connected = False # Consider if you want to automatically set to False on any write error
            self._update_plc_gui_status(f"Write Fail", "red")

    def process_measurements_for_realtime(self):
        if not self.ball_detected or self.ball_pt1 is None or len(self.field_pts)!=4: return None
        src=np.float32(self.field_pts)
        dst=np.float32([[0,0],[0,self.FIELD_H_CM],[self.FIELD_W_CM,0],[self.FIELD_W_CM,self.FIELD_H_CM]])
        H_mat = None
        try: H_mat = cv2.getPerspectiveTransform(src,dst)
        except Exception: return None
        if H_mat is None: return None
        
        ball1_pix_np = np.float32([self.ball_pt1]).reshape(-1,1,2) 
        ball1_cm_t = cv2.perspectiveTransform(ball1_pix_np,H_mat)
        if ball1_cm_t is None or ball1_cm_t.size==0: return None
        x1cm,y1cm = ball1_cm_t[0,0]

        try: x2cm,y2cm = float(self.target_x_cm_str.get()), float(self.target_y_cm_str.get())
        except ValueError: x2cm,y2cm = 0.0,0.0 
            
        dx_cm,dy_cm = x2cm-x1cm, y2cm-y1cm
        dist_cm = math.sqrt(dx_cm**2 + dy_cm**2)
        ang_rad = math.atan2(dx_cm,dy_cm) 
        ang_deg = math.degrees(ang_rad)
        if ang_deg < 0: ang_deg += 360.0
        
        return {"x1_cm":x1cm,"y1_cm":y1cm,"x2_cm":x2cm,"y2_cm":y2cm,
                "distance_cm":dist_cm,"angle_degrees":ang_deg,
                "ball1_original_px":self.ball_pt1,"H_matrix":H_mat}

    def update_measurement_display(self, data_dict): 
        if data_dict and isinstance(data_dict, dict):
            dist_disp = data_dict.get("distance_cm_final", data_dict.get("distance_cm"))
            ang_disp = data_dict.get("angle_degrees_final", data_dict.get("angle_degrees"))

            if dist_disp is not None: self.distance_display_str.set(f"Dist: {dist_disp:.1f} cm")
            else: self.distance_display_str.set("Dist: Error")
            if ang_disp is not None: self.angle_display_str.set(f"Angle: {ang_disp:.1f}°")
            else: self.angle_display_str.set("Angle: Error")
        else: self.update_measurement_display_default()

    def update_measurement_display_default(self):
        self.distance_display_str.set("Dist: N/A"); self.angle_display_str.set("Angle: N/A")

    def show_detailed_results_window(self, measurement_data):
        if not measurement_data or not isinstance(measurement_data,dict): return
        x1,y1,x2,y2,dist,angle,ball1_px,H_mat = (measurement_data.get(k) for k in ["x1_cm","y1_cm","x2_cm","y2_cm","distance_cm","angle_degrees","ball1_original_px","H_matrix"])
        if any(v is None for v in [x1,y1,x2,y2,dist,angle,ball1_px,H_mat]): messagebox.showerror("Results Data","Incomplete data."); return
        
        warp_w,warp_h=int(self.FIELD_W_CM),int(self.FIELD_H_CM)
        max_dim=500; scale_f=1.0 
        if warp_w<=0 or warp_h<=0: messagebox.showerror("Field Err","Field dims must be positive."); return
        if warp_w>max_dim or warp_h>max_dim: scale_f=min(max_dim/warp_w,max_dim/warp_h)
        disp_w,disp_h=max(1,int(warp_w*scale_f)),max(1,int(warp_h*scale_f))
        if self.img is None: messagebox.showerror("Image Err","Original image not available."); return
        
        warped_nat = cv2.warpPerspective(self.img,H_mat,(warp_w,warp_h))
        if warped_nat is None or warped_nat.size==0: messagebox.showerror("Warp Err","Failed to warp image."); return
        
        pt1d=(int(x1*scale_f),int(y1*scale_f)); pt2d=(int(x2*scale_f),int(y2*scale_f))
        warped_disp = cv2.resize(warped_nat,(disp_w,disp_h),interpolation=cv2.INTER_LINEAR)
        cv2.circle(warped_disp,pt1d,max(2,int(5*scale_f)),(0,0,255),-1) 
        cv2.putText(warped_disp,"Jack",(pt1d[0]+3,pt1d[1]-3),cv2.FONT_HERSHEY_SIMPLEX,0.4*scale_f,(255,255,255),max(1,int(1*scale_f)))
        cv2.circle(warped_disp,pt2d,max(2,int(5*scale_f)),(0,255,0),-1) 
        cv2.putText(warped_disp,"T",(pt2d[0]+3,pt2d[1]-3),cv2.FONT_HERSHEY_SIMPLEX,0.4*scale_f,(255,255,255),max(1,int(1*scale_f)))
        cv2.arrowedLine(warped_disp,pt1d,pt2d,(255,255,255),max(1,int(1*scale_f)))

        res_win=Toplevel(self.root); res_win.title("Measurement Details"); res_win.grab_set()
        res_win.top_view_photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(warped_disp,cv2.COLOR_BGR2RGB)))
        Label(res_win,text="Field Top View (Scaled)",font=("Arial",10)).pack(pady=3)
        Label(res_win,image=res_win.top_view_photo).pack(pady=5,padx=5)
        
        res_f=Frame(res_win); res_f.pack(pady=5,padx=5,fill=X)
        sfont=("Arial",9); sfont_bold = ("Arial", 9, "bold")
        Label(res_f,text=f"P1 (Jack) Original Pixel: {ball1_px}",font=sfont).pack(anchor=W)
        Label(res_f,text=f"P1 Field Coords (cm): ({x1:.1f}, {y1:.1f})",font=sfont_bold).pack(anchor=W)
        Label(res_f,text=f"Target Field Coords (cm): ({x2:.1f}, {y2:.1f})",font=sfont).pack(anchor=W)
        ttk.Separator(res_f,orient="horizontal").pack(fill="x",pady=3)
        Label(res_f,text=f"Distance to Target: {dist:.1f} cm",font=sfont_bold).pack(anchor=W)
        Label(res_f,text=f"Angle to Target (from +Y, CW): {angle:.1f}°",font=sfont_bold).pack(anchor=W)
        Button(res_win,text="Close",command=res_win.destroy,width=8).pack(pady=5)
        res_win.resizable(False,False)

    def open_calibration_window(self):
        if self.img is None: messagebox.showwarning("Calibration","Open image/camera first."); return
        if self.calibration_window_open and self.calibration_window:
            try:
                if self.calibration_window.winfo_exists():
                    self.calibration_window.lift(); self.calibration_window.refresh_all_mask_displays(); return
                else: self.calibration_window_open=False; self.calibration_window=None
            except TclError: self.calibration_window_open=False; self.calibration_window=None
        self.calibration_window = CalibrationWindow(self); self.calibration_window_open=True

    def initiate_hsv_color_pick_for_params(self, color_name):
        if self.img is None: messagebox.showwarning("Pick Color",f"Open image/camera to pick {color_name}."); return
        if self.STATUS_CLICK_COLOR_INFO_MODE: self.toggle_info_color_pick_mode() 
        
        current_picking = getattr(self, "picking_hsv_for_color", None)
        if current_picking == color_name: 
            self.canvas.config(cursor=""); self.picking_hsv_for_color = None; return
            
        self.picking_hsv_for_color = color_name; self.canvas.config(cursor="plus")
        messagebox.showinfo("Pick Color for Params",f"Click on a {color_name} ball to set its HSV params. Click '{color_name.capitalize()}' button again to cancel.")

    def process_hsv_color_pick(self, event):
        color_name_picked = getattr(self, "picking_hsv_for_color", None)
        if self.preview is None or color_name_picked is None:
            if color_name_picked: pass 
            self.canvas.config(cursor=""); self.picking_hsv_for_color=None; return

        self.canvas.config(cursor=""); self.picking_hsv_for_color=None 
        if not (0<=event.x<self.preview.shape[1] and 0<=event.y<self.preview.shape[0]): messagebox.showwarning("Pick Error","Clicked outside preview."); return
        Xo,Yo=int(event.x/self.PREVIEW_SCALE),int(event.y/self.PREVIEW_SCALE)
        if not (0<=Xo<self.img.shape[1] and 0<=Yo<self.img.shape[0]): messagebox.showwarning("Pick Error","Clicked outside original image."); return

        patch_sz=self.color_pick_patch_size_var.get(); patch_sz=max(3,patch_sz+(1 if patch_sz%2==0 else 0))
        half_p=patch_sz//2
        ys,ye=max(0,Yo-half_p),min(self.img.shape[0],Yo+half_p+1)
        xs,xe=max(0,Xo-half_p),min(self.img.shape[1],Xo+half_p+1)
        bgr_patch=self.img[ys:ye,xs:xe]
        if bgr_patch.size==0: messagebox.showwarning("Pick Error","Selected patch empty."); return
        
        hsv_patch=cv2.cvtColor(bgr_patch,cv2.COLOR_BGR2HSV)
        hp,sp,vp = int(np.median(hsv_patch[:,:,0])),int(np.median(hsv_patch[:,:,1])),int(np.median(hsv_patch[:,:,2]))
        
        target_params = self.active_detection_params["colors"][color_name_picked]
        h_del,s_del,v_del = (10 if color_name_picked=="red" else 15), 70, 70 
        s_min_th,v_min_th = 50,50 

        if color_name_picked == "white":
            picked_hsv_lower = np.array([0, 0, max(100, vp - 60)]) 
            picked_hsv_upper = np.array([179, min(80, sp + 50), 255]) 
            target_params["hsv_ranges"] = [(picked_hsv_lower, picked_hsv_upper)]
        elif color_name_picked == "red":
            ranges = []
            lh1, uh1 = hp - h_del, hp + h_del
            ls, us = max(s_min_th, sp-s_del), min(255, sp+s_del)
            lv, uv = max(v_min_th, vp-v_del), min(255, vp+v_del)
            if lh1 < 0: 
                ranges.append((np.array([max(0,179+lh1),ls,lv]), np.array([179,us,uv])))
                ranges.append((np.array([0,ls,lv]), np.array([min(179,uh1),us,uv])))
            elif uh1 > 179: 
                ranges.append((np.array([max(0,lh1),ls,lv]), np.array([179,us,uv])))
                ranges.append((np.array([0,ls,lv]), np.array([min(179,uh1-179),us,uv])))
            else: 
                ranges.append((np.array([max(0,lh1),ls,lv]), np.array([min(179,uh1),us,uv])))
            target_params["hsv_ranges"] = ranges
        else: 
            lh,uh=max(0,hp-h_del),min(179,hp+h_del)
            ls,us=max(s_min_th,sp-s_del),min(255,sp+s_del)
            lv,uv=max(v_min_th,vp-v_del),min(255,vp+v_del)
            target_params["hsv_ranges"]=[(np.array([lh,ls,lv]),np.array([uh,us,uv]))]
            
        messagebox.showinfo("Pick Success",f"HSV for '{color_name_picked}' updated. Open Adv. Calib to fine-tune.")
        if self.calibration_window_open and self.calibration_window and self.calibration_window.winfo_exists():
            self.calibration_window.calib_params["colors"][color_name_picked]=copy.deepcopy(target_params)
            self.calibration_window.load_params_to_ui()
            self.calibration_window.refresh_mask_display(color_name_picked); self.calibration_window.lift()
        if self.img is not None: self.run_full_detection_cycle(False)

    def reset_state_for_new_image_or_camera(self):
        self.field_pts=[]; self.ball_pt1=None; self.ball_all=[]; self.ball_detected=False
        if hasattr(self,"ball_status_str"): self.ball_status_str.set("Ball: Not detected")
        if hasattr(self,"corner_listbox"): self.corner_listbox.delete(0,END)
        if hasattr(self,"capture_btn") and self.capture_btn.winfo_exists():
            self.capture_btn.config(state=NORMAL if self.camera_mode else DISABLED)
        self.update_measurement_display_default(); self.has_last_known_plc_data=False
        self.sent_last_data_after_disappearance=False
        self.current_hsv_combined_mask_display = self._generate_hsv_combined_mask_for_display()
        self.picked_color_info_list = [] 

    def update_loupe_and_coords(self, event):
        if self.img is None or self.preview is None or self.preview.size==0:
            if hasattr(self.coord_label,"winfo_exists"): self.coord_label.config(text="Cursor: X: -, Y: -")
            if hasattr(self.loupe_label,"winfo_exists") and self.loupe_label.winfo_exists():
                if not hasattr(self, "_blank_loupe_photo_ref"): 
                    _b_loupe = Image.new("RGB",(self.LOUPE_DIM,self.LOUPE_DIM),"lightgrey") # Corrected color
                    self._blank_loupe_photo_ref = ImageTk.PhotoImage(image=_b_loupe)
                self.loupe_label.config(image=self._blank_loupe_photo_ref)
            return

        prev_h,prev_w=self.preview.shape[:2]
        if 0<=event.x<prev_w and 0<=event.y<prev_h:
            self.cursor_preview=(event.x,event.y)
            if hasattr(self.coord_label,"winfo_exists"): self.coord_label.config(text=f"Cursor: X:{event.x}, Y:{event.y}")
            
            Xo,Yo=int(event.x/self.PREVIEW_SCALE),int(event.y/self.PREVIEW_SCALE)
            if not(0<=Xo<self.img.shape[1] and 0<=Yo<self.img.shape[0]): return 

            h_loupe_o = int(self.LOUPE_DIM/(2*self.LOUPE_SCALE)) 
            x1,y1=max(0,Xo-h_loupe_o),max(0,Yo-h_loupe_o)
            x2,y2=min(self.img.shape[1],Xo+h_loupe_o),min(self.img.shape[0],Yo+h_loupe_o)
            patch = self.img[y1:y2,x1:x2]
            if patch.size==0: patch=np.full((self.LOUPE_DIM//int(self.LOUPE_SCALE),self.LOUPE_DIM//int(self.LOUPE_SCALE),3),128,dtype=np.uint8) 

            loupe_res = cv2.resize(patch,(self.LOUPE_DIM,self.LOUPE_DIM),interpolation=cv2.INTER_NEAREST)
            cv2.circle(loupe_res,(self.LOUPE_DIM//2,self.LOUPE_DIM//2),self.TARGET_RADIUS,self.TARGET_COLOR,-1)
            cv2.rectangle(loupe_res,(0,0),(self.LOUPE_DIM-1,self.LOUPE_DIM-1),self.LOUPE_BORDER_COLOR,self.LOUPE_BORDER)
            
            self.loupe_photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(loupe_res,cv2.COLOR_BGR2RGB)))
            if hasattr(self.loupe_label,"winfo_exists") and self.loupe_label.winfo_exists():
                self.loupe_label.config(image=self.loupe_photo) 
        else: 
            if hasattr(self.coord_label,"winfo_exists"): self.coord_label.config(text="Cursor: X: -, Y: -")

    def on_closing(self):
        print("Closing application...")
        self.running=False; self.plc_attempt_reconnect=False
        self.stop_camera_if_running() 
        if self.pymc3e and self.plc_connected:
            try: self.pymc3e.close()
            except Exception as e_plc_close: print(f"Err closing PLC: {e_plc_close}")
        self.plc_connected=False
        if self.calibration_window_open and self.calibration_window:
            try:
                if self.calibration_window.winfo_exists(): self.calibration_window.destroy()
            except: pass
        self.calibration_window=None; self.calibration_window_open=False
        if hasattr(self.root,"destroy") and self.root.winfo_exists(): self.root.destroy()
        print("Application closed.")

    def start_red_team_select_mode(self):
        self.selecting_red_team_point = True
        messagebox.showinfo("Red Team Cmd", "Click on image to select Red Team target point.")
        self.canvas.config(cursor="target") 

    def red_team(self): 
        if not self.pymc3e or not self.plc_connected:
            messagebox.showerror("PLC Error", "PLC not connected. Cannot send Red Team command."); return
        if not self.red_team_selected_point:
            messagebox.showwarning("Red Team Cmd", "Red Team target point not selected on image."); return
        if len(self.field_pts) != 4:
            messagebox.showwarning("Red Team Cmd", "Field corners not defined. Cannot transform point."); return
        
        try:
            src_pts=np.float32(self.field_pts)
            dst_pts=np.float32([[0,0],[0,self.FIELD_H_CM],[self.FIELD_W_CM,0],[self.FIELD_W_CM,self.FIELD_H_CM]])
            H_matrix = cv2.getPerspectiveTransform(src_pts,dst_pts)
            if H_matrix is None: raise ValueError("Failed to get perspective transform.")

            rt_pixel_x, rt_pixel_y = self.red_team_selected_point
            rt_pixel_np = np.float32([[rt_pixel_x, rt_pixel_y]]).reshape(-1,1,2)
            rt_cm_transformed = cv2.perspectiveTransform(rt_pixel_np, H_matrix)
            if rt_cm_transformed is None or rt_cm_transformed.size==0: raise ValueError("Failed to transform Red Team point.")
            x_rt_cm, y_rt_cm = rt_cm_transformed[0,0]

            robot_x_cm = float(self.target_x_cm_str.get())
            robot_y_cm = float(self.target_y_cm_str.get())
            
            delta_x = x_rt_cm - robot_x_cm
            delta_y = y_rt_cm - robot_y_cm 
            
            distance_to_rt_pt = math.sqrt(delta_x**2 + delta_y**2)
            angle_to_rt_pt_rad = math.atan2(delta_x, delta_y) 
            angle_to_rt_pt_deg = math.degrees(angle_to_rt_pt_rad)
            if angle_to_rt_pt_deg < 0: angle_to_rt_pt_deg += 360.0

            swing_plc = (distance_to_rt_pt * 24.096) + 5900 
            swing_plc = max(0, min(swing_plc, 23000)) 
            release_plc = 800 
            
            print(f"Red Team Cmd: Dist Robot to RT_Point: {distance_to_rt_pt:.1f} cm, Angle: {angle_to_rt_pt_deg:.1f} deg")
            print(f"Sending to PLC - Angle (D120): {angle_to_rt_pt_deg:.0f}, Swing (D106): {swing_plc:.0f}, Release (D108): {release_plc:.0f}")
            
            self.pymc3e.randomwrite(
                word_devices=["D120", "D106", "D108"],
                word_values=[self._float_to_rounded_int_word_list(angle_to_rt_pt_deg)[0],
                               self._float_to_rounded_int_word_list(swing_plc)[0],
                               self._float_to_rounded_int_word_list(release_plc)[0]],
            )
            self._update_plc_gui_status("Red Cmd Sent","cyan")
            messagebox.showinfo("Red Team Cmd",f"Red Team command sent to PLC. Angle:{angle_to_rt_pt_deg:.1f}, Swing:{swing_plc:.0f}")

        except ValueError as ve: messagebox.showerror("Red Team Cmd Error", f"Input error: {str(ve)}")
        except Exception as e_rt:
            messagebox.showerror("Red Team Cmd Error", f"Failed to send Red Team command: {str(e_rt)}")
            traceback.print_exc(); self._update_plc_gui_status("Red Cmd Fail", "orange")

if __name__ == "__main__":
    root = Tk()
    root.geometry("1000x880") # Adjusted Width x Height for screen fit
    app = FieldMeasureApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
# Author: Aapo Yla-Autio
# E-mail: aapo.yla-autio@tuni.fi
# Release: 1.2
# Release date: 16/10/2023
import cv2
import tkinter as tk
import numpy as np
import json
from tkinter import ttk, Entry
from PIL import Image, ImageTk


class WebcamViewerApp:
    def __init__(self, root):
        self.filename = 'parameters.json'
        self.config = {}

        try:
            with open(self.filename, 'r') as config_file:
                self.config = json.load(config_file)
        except (FileNotFoundError, json.JSONDecodeError):
            print("json file not found")

        self.root = root
        self.root.title(self.config['window title'])
        self.root.geometry(self.config['window dimensions'])
        self.std_deviation = self.config['std']
        self.contour_color = self.config['contour color']
        self.y_value = self.config['y value 1']
        self.y_value2 = self.config['y value 2']
        self.avg_color = self.config['avg color']
        self.median_color = self.config['median color']
        self.k_value = self.config['k value']
        self.blur_repetition = self.config['blur repetition']
        self.lower_color = np.array(self.config['lower color'])
        self.upper_color = np.array(self.config['upper color'])
        self.threshold1 = self.config['threshold 1']
        self.threshold2 = self.config['threshold 2']
        self.is_view_flipped = self.config['flipped view']

        self.contours = []
        self.selected_region = None
        self.min_r = 0
        self.min_g = 0
        self.min_b = 0
        self.max_r = 0
        self.max_g = 0
        self.max_b = 0

        # Create a frame to hold the webcam feed and buttons
        self.frame = ttk.Frame(self.root)
        self.frame.grid(row=0, column=0)

        # Create a label for the webcam feed
        self.label = ttk.Label(self.frame)
        self.label.pack()

        # Create buttons on the right side
        self.button_frame = ttk.Frame(self.root)
        self.button_frame.grid(row=0, column=1)
        self.calculate_avg = True
        self.calculate_median = False
        button_width = 7
        self.central_tendency_line_color = self.avg_color

        self.contours_label = ttk.Label(self.button_frame, text="Contours':")
        self.contours_label.grid(row=0, column=0, pady=1)
        self.avg_button = tk.Button(self.button_frame, text="Avg", command=self.change_to_avg, width=button_width,
                                    bg="red")
        self.avg_button.grid(row=0, column=1, pady=1)
        self.median_button = tk.Button(self.button_frame, text="Median", command=self.change_to_median,
                                       width=button_width)
        self.median_button.grid(row=0, column=2, pady=1)

        self.count_label = ttk.Label(self.button_frame, text="Count as:")
        self.count_label.grid(row=1, column=0, pady=1)
        self.calculate_unit = False
        self.calculate_separate = True
        self.unit_button = tk.Button(self.button_frame, text="Unit", command=self.change_to_unit, width=button_width)
        self.unit_button.grid(row=1, column=1, pady=1)
        self.separate_button = tk.Button(self.button_frame, text="Separate", command=self.change_to_separate,
                                         width=button_width, bg="green")
        self.separate_button.grid(row=1, column=2, pady=1)

        # Create a label and buttons for changing the Y value
        self.y_label = ttk.Label(self.button_frame, text="Y Value:")
        self.y_label.grid(row=2, column=0, pady=1)
        self.y_entry = Entry(self.button_frame, width=button_width)
        self.y_entry.grid(row=2, column=1, pady=1)
        self.y_entry.insert(0, str(self.y_value))
        self.y_up_button = ttk.Button(self.button_frame, text="˄", command=self.increase_y_value, width=2)
        self.y_up_button.grid(row=2, column=3, pady=1)
        self.y_down_button = ttk.Button(self.button_frame, text="˅", command=self.decrease_y_value, width=2)
        self.y_down_button.grid(row=2, column=4, pady=1)
        self.change_y_button = ttk.Button(self.button_frame, text="Change", command=self.update_y_value,
                                          width=button_width)
        self.change_y_button.grid(row=2, column=2, pady=1)

        self.y2_label = ttk.Label(self.button_frame, text="Y2 Value:")
        self.y2_label.grid(row=3, column=0, pady=1)
        self.y2_entry = Entry(self.button_frame, width=button_width)
        self.y2_entry.grid(row=3, column=1, pady=1)
        self.y2_entry.insert(0, str(self.y_value2))
        self.y2_up_button = ttk.Button(self.button_frame, text="˄", command=self.increase_y2_value, width=2)
        self.y2_up_button.grid(row=3, column=3, pady=1)
        self.y2_down_button = ttk.Button(self.button_frame, text="˅", command=self.decrease_y2_value, width=2)
        self.y2_down_button.grid(row=3, column=4, pady=1)
        self.change_y2_button = ttk.Button(self.button_frame, text="Change", command=self.update_y2_value,
                                           width=button_width)
        self.change_y2_button.grid(row=3, column=2, pady=1)

        self.k_label = ttk.Label(self.button_frame, text="K Value:")
        self.k_label.grid(row=4, column=0, pady=1)
        self.k_entry = Entry(self.button_frame, width=button_width)
        self.k_entry.grid(row=4, column=1, pady=1)
        self.k_entry.insert(0, str(self.k_value))
        self.change_k_button = ttk.Button(self.button_frame, text="Change", command=self.update_k, width=button_width)
        self.change_k_button.grid(row=4, column=2, pady=1)

        self.std_label = ttk.Label(self.button_frame, text="STD:")
        self.std_label.grid(row=5, column=0, pady=1)
        self.std_entry = Entry(self.button_frame, width=button_width)
        self.std_entry.grid(row=5, column=1, pady=1)
        self.std_entry.insert(0, str(self.std_deviation))
        self.change_std_button = ttk.Button(self.button_frame, text="Change", command=self.update_std,
                                            width=button_width)
        self.change_std_button.grid(row=5, column=2, pady=1)

        self.blur_label = ttk.Label(self.button_frame, text="Blur reps:")
        self.blur_label.grid(row=6, column=0, pady=1)
        self.blur_entry = Entry(self.button_frame, width=button_width)
        self.blur_entry.grid(row=6, column=1, pady=1)
        self.blur_entry.insert(0, str(self.blur_repetition))
        self.change_blur_button = ttk.Button(self.button_frame, text="Change", command=self.update_blur_reps,
                                             width=button_width)
        self.change_blur_button.grid(row=6, column=2, pady=1)

        self.is_blurred = False
        self.blurred_label = ttk.Label(self.button_frame, text="Blur:")
        self.blurred_label.grid(row=7, column=0, pady=1)
        self.blurred_true_button = tk.Button(self.button_frame, text="True", command=self.blurred_true,
                                             width=button_width)
        self.blurred_true_button.grid(row=7, column=1, pady=1)
        self.blurred_false_button = tk.Button(self.button_frame, text="False", command=self.blurred_false,
                                              width=button_width, bg="green")
        self.blurred_false_button.grid(row=7, column=2, pady=1)

        self.canny_label = ttk.Label(self.button_frame, text="Lower Color:")
        self.canny_label.grid(row=8, column=0, pady=1)
        self.lower_r_entry = Entry(self.button_frame, width=button_width)
        self.lower_r_entry.grid(row=8, column=1, pady=1)
        self.lower_r_entry.insert(0, str(self.lower_color[0]))
        self.lower_g_entry = Entry(self.button_frame, width=button_width)
        self.lower_g_entry.grid(row=8, column=2, pady=1)
        self.lower_g_entry.insert(0, str(self.lower_color[1]))
        self.lower_b_entry = Entry(self.button_frame, width=button_width)
        self.lower_b_entry.grid(row=8, column=3, pady=1)
        self.lower_b_entry.insert(0, str(self.lower_color[2]))
        self.change_lower_color_button = ttk.Button(self.button_frame, text="Change", command=self.update_lower_color,
                                                    width=button_width)
        self.change_lower_color_button.grid(row=8, column=4, pady=1)

        self.canny_label = ttk.Label(self.button_frame, text="Upper Color:")
        self.canny_label.grid(row=9, column=0, pady=1)
        self.upper_r_entry = Entry(self.button_frame, width=button_width)
        self.upper_r_entry.grid(row=9, column=1, pady=1)
        self.upper_r_entry.insert(0, str(self.upper_color[0]))
        self.upper_g_entry = Entry(self.button_frame, width=button_width)
        self.upper_g_entry.grid(row=9, column=2, pady=1)
        self.upper_g_entry.insert(0, str(self.upper_color[1]))
        self.upper_b_entry = Entry(self.button_frame, width=button_width)
        self.upper_b_entry.grid(row=9, column=3, pady=1)
        self.upper_b_entry.insert(0, str(self.upper_color[2]))
        self.change_upper_color_button = ttk.Button(self.button_frame, text="Change", command=self.update_upper_color,
                                                    width=button_width)
        self.change_upper_color_button.grid(row=9, column=4, pady=1)

        self.select_color = False
        self.color_selection_on = False
        self.select_color_label = ttk.Label(self.button_frame, text="Select colors")
        self.select_color_label.grid(row=10, column=0, pady=1)
        self.select_color_true_button = tk.Button(self.button_frame, text="True", command=self.select_color_true,
                                                  width=button_width)
        self.select_color_true_button.grid(row=10, column=1, pady=1)
        self.select_color_false_button = tk.Button(self.button_frame, text="False", command=self.select_color_false,
                                                   width=button_width, bg="green")
        self.select_color_false_button.grid(row=10, column=2, pady=1)

        self.blur_certain_colors = True
        self.focus_color_label = ttk.Label(self.button_frame, text="Focus colors:")
        self.focus_color_label.grid(row=11, column=0, pady=1)
        self.focus_color_true_button = tk.Button(self.button_frame, text="True", command=self.focus_color_true,
                                                 width=button_width)
        self.focus_color_true_button.grid(row=11, column=1, pady=1)
        self.focus_color_false_button = tk.Button(self.button_frame, text="False", command=self.focus_color_false,
                                                  width=button_width, bg="green")
        self.focus_color_false_button.grid(row=11, column=2, pady=1)

        self.canny_label = ttk.Label(self.button_frame, text="Canny:")
        self.canny_label.grid(row=12, column=0, pady=1)
        self.thres1_label = ttk.Label(self.button_frame, text="Threshold1:")
        self.thres1_label.grid(row=13, column=0, pady=1)
        self.thres1_entry = Entry(self.button_frame, width=button_width)
        self.thres1_entry.grid(row=13, column=1, pady=1)
        self.thres1_entry.insert(0, str(self.threshold1))
        self.change_thres1_button = ttk.Button(self.button_frame, text="Change", command=self.update_thres1,
                                               width=button_width)
        self.change_thres1_button.grid(row=13, column=2, pady=1)
        self.thres2_label = ttk.Label(self.button_frame, text="Threshold2:")
        self.thres2_label.grid(row=14, column=0, pady=1)
        self.thres2_entry = Entry(self.button_frame, width=button_width)
        self.thres2_entry.grid(row=14, column=1, pady=1)
        self.thres2_entry.insert(0, str(self.threshold2))
        self.change_thres2_button = ttk.Button(self.button_frame, text="Change", command=self.update_thres2,
                                               width=button_width)
        self.change_thres2_button.grid(row=14, column=2, pady=1)

        self.flip_view_button = ttk.Button(self.button_frame, text="Flip view", command=self.flip_view,
                                           width=button_width)
        self.flip_view_button.grid(row=15, column=0, pady=1)

        self.save = False
        self.save_label = ttk.Label(self.button_frame, text="Save mode:")
        self.save_label.grid(row=16, column=0, pady=1)
        self.save_true_button = tk.Button(self.button_frame, text="True", command=self.save_true, width=button_width)
        self.save_true_button.grid(row=16, column=1, pady=1)
        self.save_false_button = tk.Button(self.button_frame, text="False", command=self.save_false,
                                           width=button_width, bg="green")
        self.save_false_button.grid(row=16, column=2, pady=1)

        # Open the webcam
        self.cap = cv2.VideoCapture(0)

        self.selection_start = None
        self.selection_end = None
        self.label.bind("<ButtonPress-1>", self.on_mouse_press)
        self.label.bind("<B1-Motion>", self.on_mouse_drag)
        self.label.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Start updating the webcam feed
        self.update()

    def save_config(self):
        if self.save:
            with open(self.filename, 'w') as config_file:
                json.dump(self.config, config_file, indent=4)

    def change_to_avg(self):
        self.calculate_avg = True
        self.calculate_median = False
        self.central_tendency_line_color = self.avg_color
        self.avg_button.config(bg="red")
        self.median_button.config(bg="SystemButtonFace")

    def change_to_median(self):
        self.calculate_avg = False
        self.calculate_median = True
        self.central_tendency_line_color = self.median_color
        self.avg_button.config(bg="SystemButtonFace")
        self.median_button.config(bg="blue")

    def change_to_unit(self):
        self.calculate_unit = True
        self.calculate_separate = False
        self.unit_button.config(bg="green")
        self.separate_button.config(bg="SystemButtonFace")

    def change_to_separate(self):
        self.calculate_unit = False
        self.calculate_separate = True
        self.unit_button.config(bg="SystemButtonFace")
        self.separate_button.config(bg="green")

    def increase_y_value(self):
        self.y_value += 1
        self.y_entry.delete(0, "end")
        self.y_entry.insert(0, str(self.y_value))

    def decrease_y_value(self):
        self.y_value -= 1
        self.y_entry.delete(0, "end")
        self.y_entry.insert(0, str(self.y_value))

    def update_y_value(self):
        try:
            self.y_value = int(self.y_entry.get())
            self.config['y value 1'] = self.y_value
            self.save_config()
        except ValueError:
            pass

    def increase_y2_value(self):
        self.y_value2 += 1
        self.y2_entry.delete(0, "end")
        self.y2_entry.insert(0, str(self.y_value2))

    def decrease_y2_value(self):
        self.y_value2 -= 1
        self.y2_entry.delete(0, "end")
        self.y2_entry.insert(0, str(self.y_value2))

    def update_y2_value(self):
        try:
            self.y_value2 = int(self.y2_entry.get())
            self.config['y value 2'] = self.y_value2
            self.save_config()
        except ValueError:
            pass

    def update_k(self):
        try:
            self.k_value = int(self.k_entry.get())
            self.config['k value'] = self.k_value
            self.save_config()
        except ValueError:
            pass

    def update_std(self):
        try:
            self.std_deviation = int(self.std_entry.get())
            self.config['std'] = self.std_deviation
            self.save_config()
        except ValueError:
            pass

    def update_blur_reps(self):
        try:
            self.blur_repetition = int(self.blur_entry.get())
            self.config['blur repetition'] = self.blur_repetition
            self.save_config()
        except ValueError:
            pass

    def update_lower_color(self):
        try:
            self.lower_color = np.array([int(self.lower_r_entry.get()), int(self.lower_g_entry.get()),
                                         int(self.lower_b_entry.get())])
            self.config['lower color'] = self.lower_color
            self.save_config()
        except ValueError:
            pass

    def update_upper_color(self):
        try:
            self.upper_color = np.array([int(self.upper_r_entry.get()), int(self.upper_g_entry.get()),
                                         int(self.upper_b_entry.get())])
            self.config['upper color'] = self.upper_color
            self.save_config()
        except ValueError:
            pass

    def update_color_with_selection(self):
        self.lower_color[0] = self.min_r
        self.lower_color[1] = self.min_g
        self.lower_color[2] = self.min_b
        self.upper_color[0] = self.max_r
        self.upper_color[1] = self.max_g
        self.upper_color[2] = self.max_b
        self.lower_r_entry.delete(0, tk.END)
        self.lower_g_entry.delete(0, tk.END)
        self.lower_b_entry.delete(0, tk.END)
        self.upper_r_entry.delete(0, tk.END)
        self.upper_g_entry.delete(0, tk.END)
        self.upper_b_entry.delete(0, tk.END)
        self.lower_r_entry.insert(0, str(self.lower_color[0]))
        self.lower_g_entry.insert(0, str(self.lower_color[1]))
        self.lower_b_entry.insert(0, str(self.lower_color[2]))
        self.upper_r_entry.insert(0, str(self.upper_color[0]))
        self.upper_g_entry.insert(0, str(self.upper_color[1]))
        self.upper_b_entry.insert(0, str(self.upper_color[2]))

    def select_color_true(self):
        self.color_selection_on = True
        self.select_color_true_button.config(bg="green")
        self.select_color_false_button.config(bg="SystemButtonFace")

    def select_color_false(self):
        self.color_selection_on = False
        self.select_color_true_button.config(bg="SystemButtonFace")
        self.select_color_false_button.config(bg="green")

    def focus_color_true(self):
        self.blur_certain_colors = True
        self.focus_color_true_button.config(bg="green")
        self.focus_color_false_button.config(bg="SystemButtonFace")

    def focus_color_false(self):
        self.blur_certain_colors = False
        self.focus_color_true_button.config(bg="SystemButtonFace")
        self.focus_color_false_button.config(bg="green")

    def update_thres1(self):
        try:
            self.threshold1 = int(self.thres1_entry.get())
            self.config['threshold 1'] = self.threshold1
            self.save_config()
        except ValueError:
            pass

    def update_thres2(self):
        try:
            self.threshold2 = int(self.thres2_entry.get())
            self.config['threshold 2'] = self.threshold2
            self.save_config()
        except ValueError:
            pass

    def flip_view(self):
        if self.is_view_flipped:
            self.is_view_flipped = False
        else:
            self.is_view_flipped = True

    def blurred_true(self):
        self.is_blurred = True
        self.blurred_true_button.config(bg="green")
        self.blurred_false_button.config(bg="SystemButtonFace")

    def blurred_false(self):
        self.is_blurred = False
        self.blurred_true_button.config(bg="SystemButtonFace")
        self.blurred_false_button.config(bg="green")

    def save_true(self):
        self.save = True
        self.save_true_button.config(bg="green")
        self.save_false_button.config(bg="SystemButtonFace")
        self.change_y_button.config(text="Save")
        self.change_y2_button.config(text="Save")
        self.change_k_button.config(text="Save")
        self.change_std_button.config(text="Save")
        self.change_blur_button.config(text="Save")
        self.change_lower_color_button.config(text="Save")
        self.change_upper_color_button.config(text="Save")
        self.change_thres1_button.config(text="Save")
        self.change_thres2_button.config(text="Save")

    def save_false(self):
        self.save = False
        self.save_true_button.config(bg="SystemButtonFace")
        self.save_false_button.config(bg="green")
        self.change_y_button.config(text="Change")
        self.change_y2_button.config(text="Change")
        self.change_k_button.config(text="Change")
        self.change_std_button.config(text="Change")
        self.change_blur_button.config(text="Change")
        self.change_lower_color_button.config(text="Change")
        self.change_upper_color_button.config(text="Change")
        self.change_thres1_button.config(text="Change")
        self.change_thres2_button.config(text="Change")

    def on_mouse_press(self, event):
        self.selection_start = (event.x, event.y)
        self.selection_end = None

    def on_mouse_drag(self, event):
        self.selection_end = (event.x, event.y)

    def on_mouse_release(self, event):
        self.selection_end = (event.x, event.y)
        if self.selection_start and self.selection_end and self.selection_start != self.selection_end:
            x1, y1 = (min(self.selection_start[0], self.selection_end[0]),
                      min(self.selection_start[1], self.selection_end[1]))
            x2, y2 = (max(self.selection_start[0], self.selection_end[0]),
                      max(self.selection_start[1], self.selection_end[1]))
            self.selected_region = ((x1, y1), (x2, y2))
            self.selection_start = None
            self.selection_end = None
            if self.color_selection_on:
                self.select_color = True

    def blur_certain_or_all_colors(self, frame):
        # Apply Gaussian blur to the entire frame
        blurred_frame = frame
        for i in range(self.blur_repetition):
            blurred_frame = cv2.GaussianBlur(blurred_frame, (self.k_value, self.k_value), self.std_deviation)

        if self.blur_certain_colors:
            # Create a mask that highlights the pixels within the specified RGB color range
            mask = cv2.inRange(frame, self.lower_color, self.upper_color)

            # Use the mask to selectively apply the original frame or blurred frame
            result_frame = cv2.bitwise_and(frame, frame, mask=mask)
            result_frame += cv2.bitwise_and(blurred_frame, blurred_frame, mask=~mask)
            return result_frame

        return blurred_frame

    def detect_and_draw_contours_on_selected_region(self, frame):
        if self.selected_region is not None:
            x1, y1 = self.selected_region[0]
            x2, y2 = self.selected_region[1]
            selected_region = frame[y1:y2, x1:x2]
            if self.select_color:
                self.select_color = False
                self.min_r = np.min(selected_region[:, :, 0])
                self.min_g = np.min(selected_region[:, :, 1])
                self.min_b = np.min(selected_region[:, :, 2])
                self.max_r = np.max(selected_region[:, :, 0])
                self.max_g = np.max(selected_region[:, :, 1])
                self.max_b = np.max(selected_region[:, :, 2])
                self.update_color_with_selection()

            gray = cv2.cvtColor(selected_region, cv2.COLOR_RGB2GRAY)
            if not self.is_blurred:
                selected_region2 = self.blur_certain_or_all_colors(selected_region)
                gray = cv2.cvtColor(selected_region2, cv2.COLOR_RGB2GRAY)
            blurred = gray
            edges = cv2.Canny(blurred, self.threshold1, self.threshold1)
            self.contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(selected_region, self.contours, -1, self.contour_color, 2)
        return frame

    def draw_central_tendencies_of_contours_y(self, frame):
        count = 0
        for contour in self.contours:
            central_tendency_y = 0
            if len(contour) == 0:
                continue
            if self.calculate_avg:
                central_tendency_y = int(np.mean([xy[0][1] for xy in contour]))
            if self.calculate_median:
                central_tendency_y = int(np.median([xy[0][1] for xy in contour]))
            cv2.line(frame, (0, central_tendency_y + self.selected_region[0][1]),
                     (frame.shape[1], central_tendency_y + self.selected_region[0][1]),
                     self.central_tendency_line_color, 1)
            difference = central_tendency_y + self.selected_region[0][1] - self.y_value
            cv2.putText(frame, f"Difference: {difference}", (10, 15 + count * 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            count = count + 1

    def draw_central_tendency_of_contours_y(self, frame):
        y_coordinates = []
        for contour in self.contours:
            y_coordinates = y_coordinates+[xy[0][1] for xy in contour]
        if len(y_coordinates) > 0:
            central_tendency_y = 0
            if self.calculate_avg:
                central_tendency_y = int(np.mean(y_coordinates))
            if self.calculate_median:
                central_tendency_y = int(np.median(y_coordinates))
            cv2.line(frame, (0, central_tendency_y + self.selected_region[0][1]),
                     (frame.shape[1], central_tendency_y + self.selected_region[0][1]),
                     self.central_tendency_line_color, 1)
            difference = central_tendency_y + self.selected_region[0][1] - self.y_value
            cv2.putText(frame, f"Difference: {difference}", (10, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def update(self):
        ret, frame = self.cap.read()
        if ret:
            if self.is_view_flipped:
                frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.is_blurred:
                frame = self.blur_certain_or_all_colors(frame)
            frame = self.detect_and_draw_contours_on_selected_region(frame)
            if self.calculate_unit:
                self.draw_central_tendency_of_contours_y(frame)
            if self.calculate_separate:
                self.draw_central_tendencies_of_contours_y(frame)
            cv2.line(frame, (0, self.y_value), (frame.shape[1], self.y_value), (255, 0, 255), 1)
            cv2.line(frame, (0, self.y_value2), (frame.shape[1], self.y_value2), (255, 255, 0), 1)
            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.label.config(image=photo)
            self.label.image = photo

        self.root.after(10, self.update)


def main():
    root = tk.Tk()
    app = WebcamViewerApp(root)
    root.mainloop()
    app.cap.release()
    cv2.destroyAllWindows()


main()

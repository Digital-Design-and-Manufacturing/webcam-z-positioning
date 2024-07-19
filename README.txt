This program detects contours on a region that is selected by dragging and dropping. Also, other features are included
to use the code adjust the printing nozzle to right printing height. You can use the purple line to adjust correct
printing level. Then you just line the avg or median line with this assumed that the camera is attached to the moving
nozzle. Another line you can use for help of detecting retraction To run this program, main.py is needed to run and
parameters.json is needed where is saved the settings to control web camera view. Note that in the line
self.cap = cv2.VideoCapture(1) you need to set right number of camera (zero will be the default camera).

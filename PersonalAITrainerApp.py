import sys
import cv2
import numpy as np
import PoseModule as pm
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QProgressBar
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer

class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initialize detector and other variables
        self.detector = pm.poseDetector(upBody=True, smooth=True, detectionCon=0.9, trackCon=0.9)
        self.count_right = 0
        self.count_left = 0
        self.dir_right = 0
        self.dir_left = 0

        # UI Elements
        self.initUI()
        self.applyStyles()

        # Start in "Stop" mode
        self.stop()

    def initUI(self):
        self.setWindowTitle('Personal AI Trainer App')
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Video display
        self.label_video = QLabel(self)
        self.label_video.setStyleSheet("background-color: #333;")  # Set background color

        # Centering the video feed
        video_layout = QHBoxLayout()
        video_layout.addStretch(1)  # Add spacer on the left
        video_layout.addWidget(self.label_video)
        video_layout.addStretch(1)  # Add spacer on the right

        main_layout.addLayout(video_layout)

        # Progress bars and labels
        progress_layout = QHBoxLayout()

        self.right_progress = QProgressBar(self)
        self.left_progress = QProgressBar(self)

        self.right_label = QLabel('Right Arm:', self)
        self.left_label = QLabel('Left Arm:', self)

        progress_layout.addWidget(self.right_label)
        progress_layout.addWidget(self.right_progress)
        progress_layout.addWidget(self.left_label)
        progress_layout.addWidget(self.left_progress)

        main_layout.addLayout(progress_layout)

        # Control buttons
        button_layout = QHBoxLayout()

        self.start_button = QPushButton('Start', self)
        self.start_button.clicked.connect(self.start)

        self.stop_button = QPushButton('Stop', self)
        self.stop_button.clicked.connect(self.stop)

        self.exit_button = QPushButton('Exit', self)
        self.exit_button.clicked.connect(self.close)

        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.exit_button)

        main_layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Timer for video capture
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def applyStyles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #333;
            }
            QLabel {
                color: #FFF;
                font-size: 16px;
                padding: 5px;
            }
            QPushButton {
                background-color: #555;
                color: #FFF;
                padding: 10px;
                border: none;
                border-radius: 5px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #777;
            }
            QProgressBar {
                border: 2px solid #FFF;
                border-radius: 5px;
                text-align: center;
                color: #FFF;
                min-width: 200px;
            }
            QProgressBar::chunk {
                background-color: #555;
                border-radius: 3px;
            }
        """)

    def start(self):
        if not hasattr(self, 'cap') or not self.cap:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Could not open video device")
        self.timer.start(20)

    def stop(self):
        self.timer.stop()
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
            self.cap = None
        self.label_video.clear()
        self.label_video.setStyleSheet("background-color: #333;")  # Set background color

    def setGradientColor(self, progressBar, percentage):
        # Define the colors for the gradient
        cold_color = QColor(255, 0, 0)  # Red for cold
        hot_color = QColor(0, 255, 0)   # Green for hot

        # Calculate the gradient color based on percentage
        gradient_color = QColor(
            int(cold_color.red() + percentage * (hot_color.red() - cold_color.red()) / 100),
            int(cold_color.green() + percentage * (hot_color.green() - cold_color.green()) / 100),
            int(cold_color.blue() + percentage * (hot_color.blue() - cold_color.blue()) / 100)
        )

        # Apply the gradient color to the progress bar
        progressBar.setStyleSheet(
            f"QProgressBar::chunk {{ background-color: {gradient_color.name()}; }}"
        )

    def update_frame(self):
        ret, img = self.cap.read()

        # Initialize per_right and per_left
        per_right = 0
        per_left = 0

        if ret:
            img = self.detector.findPose(img, False)
            lmList = self.detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # Right Arm
                angle_right = self.detector.findAngle(img, 12, 14, 16)
                per_right = np.interp(angle_right, (35, 165), (100, 0))

                # Check for rep completion for right arm
                if per_right >= 99:  # Close to 100
                    if self.dir_right == 0:
                        self.count_right += 0.5
                        self.dir_right = 1
                elif per_right <= 1:  # Close to 0
                    if self.dir_right == 1:
                        self.count_right += 0.5
                        self.dir_right = 0

                # Left Arm
                angle_left = self.detector.findAngle(img, 11, 13, 15)
                per_left = np.interp(angle_left, (35, 165), (100, 0))

                # Check for rep completion for left arm
                if per_left >= 99:  # Close to 100
                    if self.dir_left == 0:
                        self.count_left += 0.5
                        self.dir_left = 1
                elif per_left <= 1:  # Close to 0
                    if self.dir_left == 1:
                        self.count_left += 0.5
                        self.dir_left = 0

            # Convert the frame to QPixmap and display it on QLabel
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channel = img.shape
            bytes_per_line = 3 * width
            q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.label_video.setPixmap(pixmap)

            # Update progress bars
            self.right_progress.setValue(int(per_right))
            self.left_progress.setValue(int(per_left))

            # Set gradient colors
            self.setGradientColor(self.right_progress, int(per_right))
            self.setGradientColor(self.left_progress, int(per_left))

            self.right_label.setText(f"Right Arm: {int(self.count_right)} reps")
            self.left_label.setText(f"Left Arm: {int(self.count_left)} reps")

    def closeEvent(self, event):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
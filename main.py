import sys
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QMessageBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO


class YoloApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLOv8 Cat & Oyuncak Detection")
        self.setFixedSize(900, 700)

        # ====== STYLE ======
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e1e;
                color: white;
                font-size: 14px;
            }

            QPushButton {
                background-color: #0078d7;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #005fa3;
            }

            QLabel {
                color: white;
            }

            QFrame {
                background-color: white;
                border-radius: 10px;
            }
        """)

        # ====== MODEL ======
        self.model = YOLO("best.pt")   # تأكدي أن best.pt بنفس المجلد
        self.image_path = None
        self.current_image = None

        # ====== IMAGE FRAME ======
        self.image_frame = QFrame()
        self.image_frame.setFixedSize(820, 460)

        self.image_label = QLabel("Load an image to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("color: black; font-size: 16px;")

        frame_layout = QVBoxLayout()
        frame_layout.addWidget(self.image_label)
        self.image_frame.setLayout(frame_layout)

        # ====== RESULT LABEL ======
        self.result_label = QLabel("Result: -")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #00ffcc;
        """)

        # ====== BUTTONS ======
        self.load_btn = QPushButton("Load Image")
        self.detect_btn = QPushButton("Detect")
        self.save_btn = QPushButton("Save Result")

        self.load_btn.clicked.connect(self.load_image)
        self.detect_btn.clicked.connect(self.detect_image)
        self.save_btn.clicked.connect(self.save_image)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.detect_btn)
        btn_layout.addWidget(self.save_btn)

        # ====== MAIN LAYOUT ======
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_frame)
        main_layout.addWidget(self.result_label)
        main_layout.addLayout(btn_layout)

        self.setLayout(main_layout)

    # ================= FUNCTIONS =================

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_path:
            self.image_path = file_path
            img = cv2.imread(file_path)
            self.current_image = img
            self.show_image(img)
            self.result_label.setText("Result: -")

    def show_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(
            self.image_frame.width(),
            self.image_frame.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(pixmap)

    def detect_image(self):
        if self.image_path is None:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return

        results = self.model(self.image_path)
        annotated = results[0].plot()
        self.current_image = annotated
        self.show_image(annotated)

        if len(results[0].boxes) > 0:
            cls_id = int(results[0].boxes.cls[0])
            conf = float(results[0].boxes.conf[0])
            class_name = self.model.names[cls_id]

            # تغيير الاسم من doll إلى oyuncak
            if class_name.lower() == "doll":
                class_name = "oyuncak"

            self.result_label.setText(
                f"Detected: {class_name.upper()} | Confidence: {conf:.2f}"
            )
        else:
            self.result_label.setText("No object detected")

    def save_image(self):
        if self.current_image is None:
            QMessageBox.warning(self, "Warning", "No image to save!")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "", "Images (*.png *.jpg *.jpeg)"
        )
        if file_name:
            cv2.imwrite(file_name, self.current_image)
            QMessageBox.information(self, "Saved", "Image saved successfully!")


# ================= RUN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloApp()
    window.show()
    sys.exit(app.exec_())

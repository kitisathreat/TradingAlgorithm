from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit, QComboBox, QSpinBox, QPushButton, QMessageBox
)
from PyQt5.QtCore import Qt

MODEL_TYPE_INFO = {
    "Simple Model": {
        "description": "Quick training with limited data (32-16-3 layers)",
        "best_for": "Small datasets, quick prototyping",
        "training_time": "Fast",
        "complexity": "Low"
    },
    "Standard Model": {
        "description": "Balanced for most use cases (128-64-32-3 layers)",
        "best_for": "General datasets, balanced performance",
        "training_time": "Moderate",
        "complexity": "Medium"
    },
    "Deep Model": {
        "description": "Quick training with limited data (32-16-3 layers)",
        "best_for": "Small datasets, quick prototyping",
        "training_time": "Fast",
        "complexity": "Low"
    },
    "LSTM Model": {
        "description": "Sequential pattern recognition (LSTM layers)",
        "best_for": "Time series, sequential data",
        "training_time": "Slower",
        "complexity": "High"
    },
    "Ensemble Model": {
        "description": "Combines multiple architectures",
        "best_for": "Robustness, diverse data",
        "training_time": "Varies",
        "complexity": "High"
    }
}

class CustomModelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model Configuration")
        self.setMinimumWidth(600)
        layout = QVBoxLayout(self)

        # Model Type
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(list(MODEL_TYPE_INFO.keys()))
        self.model_type_combo.currentTextChanged.connect(self.update_model_info)
        type_layout.addWidget(self.model_type_combo)
        layout.addLayout(type_layout)

        # Model Name, Author, Epochs
        row_layout = QHBoxLayout()
        row_layout.addWidget(QLabel("Model Name:"))
        self.model_name_edit = QLineEdit()
        row_layout.addWidget(self.model_name_edit)
        row_layout.addWidget(QLabel("Author:"))
        self.author_edit = QLineEdit()
        row_layout.addWidget(self.author_edit)
        row_layout.addWidget(QLabel("Minimum Training Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(20)
        row_layout.addWidget(self.epochs_spin)
        layout.addLayout(row_layout)

        # Description, Best for, Training time, Complexity
        self.desc_label = QLabel()
        self.best_for_label = QLabel()
        self.training_time_label = QLabel()
        self.complexity_label = QLabel()
        layout.addWidget(self.desc_label)
        layout.addWidget(self.best_for_label)
        layout.addWidget(self.training_time_label)
        layout.addWidget(self.complexity_label)

        # Training Subjects
        layout.addWidget(QLabel("Training Subjects (one per line):"))
        self.subjects_edit = QTextEdit()
        self.subjects_edit.setPlaceholderText("e.g., 'John Doe', 'Trading Team A', 'Myself'")
        layout.addWidget(self.subjects_edit)

        # Error label
        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: red;")
        layout.addWidget(self.error_label)

        # Buttons
        btn_layout = QHBoxLayout()
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.ok_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        self.update_model_info(self.model_type_combo.currentText())

    def update_model_info(self, model_type):
        info = MODEL_TYPE_INFO.get(model_type, {})
        self.desc_label.setText(f"<b>Description:</b> {info.get('description', '')}")
        self.best_for_label.setText(f"<b>Best for:</b> {info.get('best_for', '')}")
        self.training_time_label.setText(f"<b>Training time:</b> {info.get('training_time', '')}")
        self.complexity_label.setText(f"<b>Complexity:</b> {info.get('complexity', '')}")

    def accept(self):
        # Validation
        if not self.model_name_edit.text().strip():
            self.error_label.setText("Please enter a model name.")
            return
        if not self.subjects_edit.toPlainText().strip():
            self.error_label.setText("Please specify at least one training subject.")
            return
        self.error_label.setText("")
        super().accept()

    def get_model_config(self):
        return {
            "model_type": self.model_type_combo.currentText(),
            "model_name": self.model_name_edit.text().strip(),
            "author": self.author_edit.text().strip(),
            "min_epochs": self.epochs_spin.value(),
            "description": self.desc_label.text(),
            "best_for": self.best_for_label.text(),
            "training_time": self.training_time_label.text(),
            "complexity": self.complexity_label.text(),
            "subjects": [s.strip() for s in self.subjects_edit.toPlainText().splitlines() if s.strip()]
        } 
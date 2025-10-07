#gui.py
import sys
import os
import subprocess
from functools import partial

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QCheckBox,
    QSpinBox, QDoubleSpinBox, QLineEdit, QPushButton,
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QVBoxLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor

# Simple translations
TRANSLATIONS = {
    'en': {
        'Train Model': 'Train Model',
        'Start Camera': 'Start Camera',
        'Camera Number': 'Camera Number',
        'Min Area': 'Min Area',
        'Face Interval': 'Face Interval',
        'Duration (s)': 'Duration (s)',
        'Snapshots': 'Snapshots',
        'Snapshot Interval': 'Snapshot Interval',
        'No Record': 'No Record',
        'No Display': 'No Display',
        'Resolution': 'Resolution',
        'FPS': 'FPS',
        'Model Path': 'Model Path',
        'Data Dir': 'Data Dir',
        'Threshold': 'Threshold',
        'Verbose': 'Verbose',
        'Theme': 'Theme',
        'Language': 'Language',
        'Run': 'Run',
        'Browse': 'Browse...'
    },
    'tr': {
        'Train Model': 'Model Eğit',
        'Start Camera': 'Kamerayı Başlat',
        'Camera Number': 'Kamera Sayısı',
        'Min Area': 'Min Alan',
        'Face Interval': 'Yüz Aralığı',
        'Duration (s)': 'Süre (s)',
        'Snapshots': 'Anlık Görüntüler',
        'Snapshot Interval': 'Anlık Gör. Aralığı',
        'No Record': 'Kaydetme Yok',
        'No Display': 'Görüntüleme Yok',
        'Resolution': 'Çözünürlük',
        'FPS': 'FPS',
        'Model Path': 'Model Yolu',
        'Data Dir': 'Veri Dizini',
        'Threshold': 'Eşik',
        'Verbose': 'Ayrıntı',
        'Theme': 'Tema',
        'Language': 'Dil',
        'Run': 'Çalıştır',
        'Browse': 'Gözat...'
    }
}

class ConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Security Cam Configuration")
        self.init_ui()

    def init_ui(self):
        # Central widget and layout
        self.central = QWidget()
        self.setCentralWidget(self.central)
        main_layout = QVBoxLayout(self.central)
        form = QFormLayout()

        # Language selection
        lang_label = QLabel(self.trans('Language'))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(['en', 'tr'])
        self.lang_combo.currentTextChanged.connect(self.update_translations)
        form.addRow(lang_label, self.lang_combo)

        # Theme selection
        theme_label = QLabel(self.trans('Theme'))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['Light', 'Dark'])
        self.theme_combo.currentTextChanged.connect(self.apply_theme)
        form.addRow(theme_label, self.theme_combo)

        # Train model flag
        train_label = QLabel(self.trans('Train Model'))
        self.chk_train = QCheckBox()
        form.addRow(train_label, self.chk_train)

        # Start camera flag
        cam_label = QLabel(self.trans('Start Camera'))
        self.chk_cam = QCheckBox()
        form.addRow(cam_label, self.chk_cam)

        # Camera number
        camnum_label = QLabel(self.trans('Camera Number'))
        self.spin_cam_num = QSpinBox()
        self.spin_cam_num.setRange(1, 8)
        form.addRow(camnum_label, self.spin_cam_num)

        # Minimum area
        minarea_label = QLabel(self.trans('Min Area'))
        self.spin_min_area = QSpinBox()
        self.spin_min_area.setRange(100, 10000)
        form.addRow(minarea_label, self.spin_min_area)

        # Face recognition interval
        faceint_label = QLabel(self.trans('Face Interval'))
        self.spin_face_int = QSpinBox()
        self.spin_face_int.setRange(1, 100)
        form.addRow(faceint_label, self.spin_face_int)

        # Recording duration
        duration_label = QLabel(self.trans('Duration (s)'))
        self.spin_duration = QSpinBox()
        self.spin_duration.setRange(10, 3600)
        form.addRow(duration_label, self.spin_duration)

        # Snapshots flag
        snaps_label = QLabel(self.trans('Snapshots'))
        self.chk_snap = QCheckBox()
        form.addRow(snaps_label, self.chk_snap)

        # Snapshot interval
        snapint_label = QLabel(self.trans('Snapshot Interval'))
        self.spin_snap_int = QSpinBox()
        self.spin_snap_int.setRange(1, 300)
        form.addRow(snapint_label, self.spin_snap_int)

        # No record flag
        norec_label = QLabel(self.trans('No Record'))
        self.chk_norec = QCheckBox()
        form.addRow(norec_label, self.chk_norec)

        # No display flag
        nodis_label = QLabel(self.trans('No Display'))
        self.chk_nodis = QCheckBox()
        form.addRow(nodis_label, self.chk_nodis)

        # Resolution
        res_label = QLabel(self.trans('Resolution'))
        self.line_res = QLineEdit('640x480')
        form.addRow(res_label, self.line_res)

        # FPS
        fps_label = QLabel(self.trans('FPS'))
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(1, 60)
        self.spin_fps.setValue(20)
        form.addRow(fps_label, self.spin_fps)

        # Model path with browse button
        model_label = QLabel(self.trans('Model Path'))
        self.line_model = QLineEdit('model.pkl')
        self.btn_model = QPushButton(self.trans('Browse'))
        self.btn_model.clicked.connect(lambda: self.browse_file(self.line_model))
        form.addRow(model_label, self.hbox(self.line_model, self.btn_model))

        # Data directory with browse button
        data_label = QLabel(self.trans('Data Dir'))
        self.line_data = QLineEdit('faces')
        self.btn_data = QPushButton(self.trans('Browse'))
        self.btn_data.clicked.connect(lambda: self.browse_dir(self.line_data))
        form.addRow(data_label, self.hbox(self.line_data, self.btn_data))

        # Threshold
        thr_label = QLabel(self.trans('Threshold'))
        self.spin_thr = QDoubleSpinBox()
        self.spin_thr.setRange(0.0, 1.0)
        self.spin_thr.setDecimals(2)
        self.spin_thr.setValue(0.7)
        form.addRow(thr_label, self.spin_thr)

        # Verbose flag
        verb_label = QLabel(self.trans('Verbose'))
        self.chk_verbose = QCheckBox()
        form.addRow(verb_label, self.chk_verbose)

        # Add form to layout
        main_layout.addLayout(form)

        # Run button
        self.btn_run = QPushButton(self.trans('Run'))
        self.btn_run.clicked.connect(self.run_program)
        main_layout.addWidget(self.btn_run)

        # Initial translations and theme
        self.update_translations(self.lang_combo.currentText())
        self.apply_theme(self.theme_combo.currentText())

    def trans(self, key):
        # Return translation based on current language
        lang = self.lang_combo.currentText() if hasattr(self, 'lang_combo') else 'en'
        return TRANSLATIONS.get(lang, TRANSLATIONS['en']).get(key, key)

    def hbox(self, *widgets):
        # Helper to wrap widgets in a horizontal layout
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for w in widgets:
            layout.addWidget(w)
        return container

    def update_translations(self, lang):
        # Update all labels and buttons to the selected language
        for label in self.findChildren(QLabel):
            text = label.text()
            for k, v in TRANSLATIONS['en'].items():
                if text == v or text == TRANSLATIONS['tr'][k]:
                    label.setText(TRANSLATIONS[lang][k])
                    break
        for button in self.findChildren(QPushButton):
            text = button.text()
            for k, v in TRANSLATIONS['en'].items():
                if text == v or text == TRANSLATIONS['tr'][k]:
                    button.setText(TRANSLATIONS[lang][k])
                    break

    def apply_theme(self, theme):
        # Apply light or dark palette
        pal = QPalette()
        if theme == 'Dark':
            pal.setColor(QPalette.Window, QColor(53, 53, 53))
            pal.setColor(QPalette.WindowText, Qt.white)
            pal.setColor(QPalette.Base, QColor(25, 25, 25))
            pal.setColor(QPalette.Text, Qt.white)
        QApplication.instance().setPalette(pal)

    def browse_file(self, line):
        path, _ = QFileDialog.getOpenFileName(self, self.trans('Browse'), os.getcwd(), "*")
        if path:
            line.setText(path)

    def browse_dir(self, line):
        path = QFileDialog.getExistingDirectory(self, self.trans('Browse'), os.getcwd())
        if path:
            line.setText(path)

    def run_program(self):
        # Collect arguments and launch CLI
        args = []
        if self.chk_train.isChecked(): args.append('-train')
        if self.chk_cam.isChecked(): args.append('-cam')
        args += ['--cam-num', str(self.spin_cam_num.value())]
        args += ['--min-area', str(self.spin_min_area.value())]
        args += ['--face-interval', str(self.spin_face_int.value())]
        args += ['--duration', str(self.spin_duration.value())]
        if self.chk_snap.isChecked(): args.append('--snapshot')
        args += ['--snapshot-interval', str(self.spin_snap_int.value())]
        if self.chk_norec.isChecked(): args.append('--no-record')
        if self.chk_nodis.isChecked(): args.append('--no-display')
        args += ['--resolution', self.line_res.text()]
        args += ['--fps', str(self.spin_fps.value())]
        args += ['--model', self.line_model.text()]
        args += ['--data', self.line_data.text()]
        args += ['--threshold', str(self.spin_thr.value())]
        if self.chk_verbose.isChecked(): args.append('--verbose')
        subprocess.Popen([sys.executable, os.path.abspath(sys.argv[0])] + args)
        self.close()


def launch_gui():
    app = QApplication(sys.argv)
    win = ConfigWindow()
    win.show()
    app.exec()

if __name__ == '__main__':
    launch_gui()
import sys
import os
import json
import cv2
import torch
import torch.nn.functional as F
import numpy as np

from torchvision import transforms
from PyQt6 import QtWidgets, QtGui, QtCore

# Import trực tiếp từ doan.py để GUI khớp model và cách tính score
from doan import (
    IMG_SIZE,
    DEVICE,
    OUTPUT_DIR,
    build_model,
    forward_model,
    PerceptualFeatureExtractor,
)


# =========================
# GUI
# =========================
class UI:
    def __init__(self):
        self.model = None
        self.metrics = None
        self.image_path = None
        self.perceptual_extractor = None

    def setup(self, window):
        window.setWindowTitle("Anomaly Detection System")
        window.resize(1280, 860)

        self.tabs = QtWidgets.QTabWidget(window)
        self.tabs.setGeometry(0, 0, 1280, 860)

        self.tab_detect = QtWidgets.QWidget()
        self.tab_chart = QtWidgets.QWidget()
        self.tab_history = QtWidgets.QWidget()

        self.tabs.addTab(self.tab_detect, "Phát hiện")
        self.tabs.addTab(self.tab_chart, "Biểu đồ")
        self.tabs.addTab(self.tab_history, "Lịch sử Train")

        self.setup_detect_tab()
        self.setup_chart_tab()
        self.setup_history_tab()

        self.combo_dataset.currentTextChanged.connect(self.load_experiment)
        self.combo_model.currentTextChanged.connect(self.load_experiment)
        self.combo_experiment.currentTextChanged.connect(self.load_experiment)

        self.combo_chart.currentTextChanged.connect(self.load_chart)

        self.btn_choose.clicked.connect(self.choose_image)
        self.btn_detect.clicked.connect(self.detect)

        self.combo_hist_dataset.currentTextChanged.connect(self.load_history)
        self.combo_hist_model.currentTextChanged.connect(self.load_history)
        self.combo_hist_exp.currentTextChanged.connect(self.load_history)
        self.btn_refresh_history.clicked.connect(self.load_history)

        self.load_experiment()
        self.load_history()

    # =========================
    # TAB DETECT
    # =========================
    def setup_detect_tab(self):
        self.combo_dataset = QtWidgets.QComboBox(self.tab_detect)
        self.combo_dataset.setGeometry(30, 20, 150, 34)
        self.combo_dataset.addItems(["hazelnut", "bottle", "cifar"])

        self.combo_model = QtWidgets.QComboBox(self.tab_detect)
        self.combo_model.setGeometry(200, 20, 150, 34)
        self.combo_model.addItems(["baseline", "advanced"])

        self.combo_experiment = QtWidgets.QComboBox(self.tab_detect)
        self.combo_experiment.setGeometry(370, 20, 220, 34)
        self.combo_experiment.addItems([
            "adam_weight_decay",
            "sgd_weight_decay",
            "adam_augmentation",
        ])

        self.btn_choose = QtWidgets.QPushButton("Chọn ảnh", self.tab_detect)
        self.btn_choose.setGeometry(620, 20, 120, 34)

        self.btn_detect = QtWidgets.QPushButton("Phát hiện", self.tab_detect)
        self.btn_detect.setGeometry(760, 20, 120, 34)

        self.img_input = QtWidgets.QLabel(self.tab_detect)
        self.img_input.setGeometry(40, 90, 340, 280)
        self.img_input.setStyleSheet("border:1px solid #aaa; background:#f0f0f0;")
        self.img_input.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_input.setText("Ảnh gốc")

        self.img_recon = QtWidgets.QLabel(self.tab_detect)
        self.img_recon.setGeometry(450, 90, 340, 280)
        self.img_recon.setStyleSheet("border:1px solid #aaa; background:#f0f0f0;")
        self.img_recon.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_recon.setText("Ảnh tái tạo")

        self.img_heat = QtWidgets.QLabel(self.tab_detect)
        self.img_heat.setGeometry(860, 90, 340, 280)
        self.img_heat.setStyleSheet("border:1px solid #aaa; background:#f0f0f0;")
        self.img_heat.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.img_heat.setText("Heatmap lỗi")

        self.result = QtWidgets.QLabel("Kết quả", self.tab_detect)
        self.result.setGeometry(180, 400, 860, 56)
        self.result.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.result.setStyleSheet(
            "border:2px solid #888; font-size:18px; font-weight:bold;"
        )

        self.info_score = QtWidgets.QLabel(
            "Pixel score: - | Feature score: - | Combined score: -",
            self.tab_detect
        )
        self.info_score.setGeometry(30, 475, 1200, 26)

        self.info_threshold = QtWidgets.QLabel("Threshold: -", self.tab_detect)
        self.info_threshold.setGeometry(30, 507, 1200, 26)

        self.info_metrics = QtWidgets.QLabel("AUROC / PR-AUC: -", self.tab_detect)
        self.info_metrics.setGeometry(30, 539, 1200, 26)

        self.info_direction = QtWidgets.QLabel(
            "Direction: higher_is_anomaly",
            self.tab_detect
        )
        self.info_direction.setGeometry(30, 571, 1200, 26)

        self.dataset_note = QtWidgets.QTextEdit(self.tab_detect)
        self.dataset_note.setGeometry(30, 610, 1200, 200)
        self.dataset_note.setReadOnly(True)

    # =========================
    # TAB CHART
    # =========================
    def setup_chart_tab(self):
        self.combo_chart = QtWidgets.QComboBox(self.tab_chart)
        self.combo_chart.setGeometry(30, 20, 240, 34)
        self.combo_chart.addItems([
            "learning_curve",
            "pixel_roc",
            "pixel_pr",
            "feature_roc",
            "feature_pr",
            "combined_roc",
            "combined_pr",
        ])

        self.chart = QtWidgets.QLabel(self.tab_chart)
        self.chart.setGeometry(30, 70, 780, 660)
        self.chart.setStyleSheet("border:1px solid #999; background:#f7f7f7;")
        self.chart.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.report = QtWidgets.QTextEdit(self.tab_chart)
        self.report.setGeometry(840, 70, 400, 660)
        self.report.setReadOnly(True)
        self.report.setStyleSheet("font-family: monospace; font-size: 12px;")

    # =========================
    # TAB HISTORY
    # =========================
    def setup_history_tab(self):
        QtWidgets.QLabel("Dataset:", self.tab_history).setGeometry(30, 20, 70, 28)
        self.combo_hist_dataset = QtWidgets.QComboBox(self.tab_history)
        self.combo_hist_dataset.setGeometry(100, 20, 130, 30)
        self.combo_hist_dataset.addItems(["hazelnut", "bottle", "cifar"])

        QtWidgets.QLabel("Model:", self.tab_history).setGeometry(250, 20, 55, 28)
        self.combo_hist_model = QtWidgets.QComboBox(self.tab_history)
        self.combo_hist_model.setGeometry(310, 20, 130, 30)
        self.combo_hist_model.addItems(["baseline", "advanced"])

        QtWidgets.QLabel("Experiment:", self.tab_history).setGeometry(460, 20, 90, 28)
        self.combo_hist_exp = QtWidgets.QComboBox(self.tab_history)
        self.combo_hist_exp.setGeometry(555, 20, 200, 30)
        self.combo_hist_exp.addItems([
            "adam_weight_decay",
            "sgd_weight_decay",
            "adam_augmentation",
        ])

        self.btn_refresh_history = QtWidgets.QPushButton("Làm mới", self.tab_history)
        self.btn_refresh_history.setGeometry(780, 18, 120, 34)

        self.history_table = QtWidgets.QTableWidget(self.tab_history)
        self.history_table.setGeometry(30, 70, 580, 700)
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels([
            "Epoch", "Train Loss", "Val Loss", "LR", "Thời gian"
        ])
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self.history_table.setAlternatingRowColors(True)

        self.history_summary = QtWidgets.QTextEdit(self.tab_history)
        self.history_summary.setGeometry(630, 70, 600, 700)
        self.history_summary.setReadOnly(True)
        self.history_summary.setStyleSheet("font-family: monospace; font-size: 12px;")

    # =========================
    # HELPERS
    # =========================
    def set_fit_pixmap(self, label, pixmap):
        label.setPixmap(
            pixmap.scaled(
                label.width(),
                label.height(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
        )

    def show_bgr(self, label, img_bgr):
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, c = rgb.shape

        qimg = QtGui.QImage(
            rgb.data,
            w,
            h,
            c * w,
            QtGui.QImage.Format.Format_RGB888,
        )

        self.set_fit_pixmap(label, QtGui.QPixmap.fromImage(qimg))

    def get_opt_reg(self):
        v = self.combo_experiment.currentText()
        parts = v.split("_", 1)
        return parts[0], parts[1] if len(parts) > 1 else v

    def get_hist_opt_reg(self):
        v = self.combo_hist_exp.currentText()
        parts = v.split("_", 1)
        return parts[0], parts[1] if len(parts) > 1 else v

    def exp_dir(self):
        ds = self.combo_dataset.currentText()
        mdl = self.combo_model.currentText()
        opt, reg = self.get_opt_reg()
        return os.path.join(OUTPUT_DIR, f"{ds}_{mdl}_{opt}_{reg}")

    def hist_exp_dir(self):
        ds = self.combo_hist_dataset.currentText()
        mdl = self.combo_hist_model.currentText()
        opt, reg = self.get_hist_opt_reg()
        return os.path.join(OUTPUT_DIR, f"{ds}_{mdl}_{opt}_{reg}")

    def fmt_float(self, value, ndigits=4):
        try:
            return f"{float(value):.{ndigits}f}"
        except Exception:
            return "-"

    def normalize_with_minmax(self, score, mn, mx):
        return (score - mn) / (mx - mn + 1e-8)

    def get_selected_score_and_threshold(self, pixel_score, feature_score, combined_score):
        selected_score = self.metrics.get("selected_score", "pixel_score")

        if selected_score == "combined_score":
            return (
                combined_score,
                float(self.metrics.get("threshold_combined", 0.0)),
                "combined",
            )

        if selected_score == "feature_score":
            return (
                feature_score,
                float(self.metrics.get("threshold_feature", 0.0)),
                "feature",
            )

        return (
            pixel_score,
            float(self.metrics.get("threshold_pixel", 0.0)),
            "pixel",
        )

    # =========================
    # LOAD EXPERIMENT
    # =========================
    def load_experiment(self):
        exp_dir = self.exp_dir()
        model_path = os.path.join(exp_dir, "model.pth")
        json_path = os.path.join(exp_dir, "metrics.json")

        if not os.path.exists(model_path) or not os.path.exists(json_path):
            self.model = None
            self.metrics = None
            self.perceptual_extractor = None

            self.result.setText("Chưa có model. Hãy chạy doan.py trước.")
            self.result.setStyleSheet(
                "border:2px solid #888; font-size:18px; font-weight:bold;"
            )
            self.report.setText(f"Chưa có kết quả tại:\n{exp_dir}")
            self.chart.clear()
            self.chart.setText("Chưa có biểu đồ")
            self.dataset_note.setText("Chưa có thông tin dataset.")
            return

        model_name = self.combo_model.currentText()

        self.model = build_model(model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()

        if model_name == "advanced":
            self.perceptual_extractor = PerceptualFeatureExtractor().to(DEVICE)
            self.perceptual_extractor.eval()
        else:
            self.perceptual_extractor = None

        with open(json_path, "r", encoding="utf-8") as f:
            self.metrics = json.load(f)

        selected_score = self.metrics.get("selected_score", "pixel_score")
        selected_method = self.metrics.get("selected_threshold_method", "val95")

        self.info_threshold.setText(
            f"Method={selected_method} | Selected={selected_score} | "
            f"pixel={self.fmt_float(self.metrics.get('threshold_pixel'), 6)} | "
            f"feature={self.fmt_float(self.metrics.get('threshold_feature'), 6)} | "
            f"combined={self.fmt_float(self.metrics.get('threshold_combined'), 6)}"
        )

        self.info_metrics.setText(
            f"AUROC pixel={self.fmt_float(self.metrics.get('auroc_pixel'))} "
            f"feature={self.fmt_float(self.metrics.get('auroc_feature'))} "
            f"combined={self.fmt_float(self.metrics.get('auroc_combined'))} | "
            f"PR-AUC pixel={self.fmt_float(self.metrics.get('prauc_pixel'))} "
            f"feature={self.fmt_float(self.metrics.get('prauc_feature'))} "
            f"combined={self.fmt_float(self.metrics.get('prauc_combined'))}"
        )

        self.info_direction.setText("Direction: higher_is_anomaly | score cao hơn = bất thường")

        self.update_dataset_note()
        self.update_report()
        self.load_chart()

    def update_dataset_note(self):
        path = os.path.join(OUTPUT_DIR, "dataset_reports.json")

        if not os.path.exists(path):
            self.dataset_note.setText("Chưa có dataset_reports.json")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                reports = json.load(f)
        except Exception as e:
            self.dataset_note.setText(f"Lỗi đọc dataset_reports.json: {e}")
            return

        ds = self.combo_dataset.currentText()
        rep = next((r for r in reports if r.get("dataset") == ds), None)

        if rep is None:
            self.dataset_note.setText("Không có thông tin dataset.")
            return

        lines = [f"Dataset: {ds}"]

        if "normal_class" in rep:
            lines.append(f"Normal class: {rep['normal_class']}")

        lines += [
            f"Train good: {rep.get('train_good', 0)}",
            f"Validation good: {rep.get('val_good', 0)}",
            f"Test total: {rep.get('test_total', 0)}",
        ]

        if rep.get("test_breakdown"):
            lines.append(
                "Test breakdown: " +
                ", ".join(f"{k}={v}" for k, v in rep["test_breakdown"].items())
            )

        lines.append("Khó khăn đặc thù:")

        for note in rep.get("notes", []):
            lines.append(f"  - {note}")

        self.dataset_note.setText("\n".join(lines))

    def update_report(self):
        if self.metrics is None:
            self.report.setText("Không có dữ liệu.")
            return

        m = self.metrics

        txt = "\n".join([
            "BÁO CÁO THÍ NGHIỆM",
            "=" * 38,
            f"Dataset       : {m.get('dataset', '-')}",
            f"Model         : {m.get('model_name', '-')}",
            f"Optimizer     : {m.get('optimizer', '-')}",
            f"Regularization: {m.get('regularization', '-')}",
            f"Số tham số    : {m.get('params', '-'):,}" if isinstance(m.get("params"), int) else f"Số tham số    : {m.get('params', '-')}",
            "",
            "── Ngưỡng chính GUI đang dùng ──",
            f"  Method   : {m.get('selected_threshold_method', '-')}",
            f"  Selected : {m.get('selected_score', '-')}",
            f"  pixel    : {self.fmt_float(m.get('threshold_pixel'), 6)}",
            f"  feature  : {self.fmt_float(m.get('threshold_feature'), 6)}",
            f"  combined : {self.fmt_float(m.get('threshold_combined'), 6)}",
            "",
            "── Ngưỡng validation p95 ──",
            f"  pixel    : {self.fmt_float(m.get('threshold_pixel_val95'), 6)}",
            f"  feature  : {self.fmt_float(m.get('threshold_feature_val95'), 6)}",
            f"  combined : {self.fmt_float(m.get('threshold_combined_val95'), 6)}",
            "",
            "── Metrics ──",
            f"  AUROC pixel     : {self.fmt_float(m.get('auroc_pixel'))}",
            f"  PR-AUC pixel    : {self.fmt_float(m.get('prauc_pixel'))}",
            f"  AUROC feature   : {self.fmt_float(m.get('auroc_feature'))}",
            f"  PR-AUC feature  : {self.fmt_float(m.get('prauc_feature'))}",
            f"  AUROC combined  : {self.fmt_float(m.get('auroc_combined'))}",
            f"  PR-AUC combined : {self.fmt_float(m.get('prauc_combined'))}",
            "",
            "── Best F1 tham khảo ──",
            f"  pixel    : {self.fmt_float(m.get('best_f1_pixel'))}",
            f"  feature  : {self.fmt_float(m.get('best_f1_feature'))}",
            f"  combined : {self.fmt_float(m.get('best_f1_combined'))}",
            "",
            "── Loss ──",
            f"  Train cuối   : {self.fmt_float(m.get('train_loss_last'), 6)}",
            f"  Val cuối     : {self.fmt_float(m.get('val_loss_last'), 6)}",
            f"  Best val loss: {self.fmt_float(m.get('best_val_loss'), 6)}",
            "",
            "── Ghi chú ──",
            "  Direction      : higher_is_anomaly",
            "  Score cao hơn  → bất thường",
            "  GUI đang dùng threshold chính trong metrics.json",
            "  Không dùng Youden/F1 làm ngưỡng chính",
        ])

        self.report.setText(txt)

    # =========================
    # DETECT
    # =========================
    def choose_image(self):
        file, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Chọn ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if file:
            self.image_path = file
            self.set_fit_pixmap(self.img_input, QtGui.QPixmap(file))

    def compute_single_scores(self, x):
        model_name = self.combo_model.currentText()

        with torch.no_grad():
            out = forward_model(self.model, model_name, x, train_mode=False)

            diff = (out - x) ** 2
            err_map = diff.squeeze().cpu().numpy().mean(axis=0)
            pixel_score = float(diff.mean().item())

            if model_name == "advanced":
                ms_w = [0.2, 0.3, 0.5]

                fx = self.model.get_multiscale_features(x)
                fo = self.model.get_multiscale_features(out)

                multi_scale_score = sum(
                    ms_w[i] * float(((fx[i + 1] - fo[i + 1]) ** 2).mean().item())
                    for i in range(3)
                )

                if self.perceptual_extractor is not None:
                    px = self.perceptual_extractor(x)
                    po = self.perceptual_extractor(out)
                    perceptual_score = float(((px - po) ** 2).mean().item())
                else:
                    perceptual_score = 0.0

                # Khớp doan.py:
                # feature_score = 0.6 * multi_scale_score + 0.4 * perceptual_score
                feature_score = 0.6 * multi_scale_score + 0.4 * perceptual_score

            else:
                fxb = self.model.get_features(x)
                fob = self.model.get_features(out)
                feature_score = float(((fxb - fob) ** 2).mean().item())

        return out, err_map, pixel_score, feature_score

    def detect(self):
        if self.model is None or self.metrics is None:
            self.result.setText("Chưa load được model.")
            return

        if self.image_path is None:
            self.result.setText("Vui lòng chọn ảnh trước.")
            return

        img = cv2.imread(self.image_path)

        if img is None:
            self.result.setText("Không đọc được ảnh.")
            return

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = transforms.ToTensor()(rgb).unsqueeze(0).to(DEVICE)

        out, err_map, pixel_score, feature_score = self.compute_single_scores(x)

        # Combined score theo min-max validation đã lưu trong metrics.json
        val_pixel_min = float(self.metrics.get("val_pixel_min", 0.0))
        val_pixel_max = float(self.metrics.get("val_pixel_max", 1.0))
        val_feature_min = float(self.metrics.get("val_feature_min", 0.0))
        val_feature_max = float(self.metrics.get("val_feature_max", 1.0))

        pixel_n = self.normalize_with_minmax(pixel_score, val_pixel_min, val_pixel_max)
        feature_n = self.normalize_with_minmax(feature_score, val_feature_min, val_feature_max)

        pixel_w = float(self.metrics.get("combined_pixel_weight", 0.4))
        feature_w = float(self.metrics.get("combined_feature_weight", 0.6))

        combined_score = pixel_w * pixel_n + feature_w * feature_n

        # Hiển thị ảnh tái tạo
        recon = out.squeeze().detach().cpu().permute(1, 2, 0).numpy()
        recon = np.clip(recon * 255, 0, 255).astype(np.uint8)
        recon_bgr = cv2.cvtColor(recon, cv2.COLOR_RGB2BGR)
        self.show_bgr(self.img_recon, recon_bgr)

        # Hiển thị heatmap
        heat = (err_map - err_map.min()) / (err_map.max() - err_map.min() + 1e-8)
        heat = cv2.GaussianBlur((heat * 255).astype(np.uint8), (11, 11), 0)
        heat_c = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        self.show_bgr(self.img_heat, heat_c)

        self.info_score.setText(
            f"Pixel score: {pixel_score:.6f} | "
            f"Feature score: {feature_score:.6f} | "
            f"Combined score: {combined_score:.6f}"
        )

        score_used, thr_used, score_name = self.get_selected_score_and_threshold(
            pixel_score,
            feature_score,
            combined_score
        )

        method = self.metrics.get("selected_threshold_method", "val95")
        auroc_value = self.metrics.get(f"auroc_{score_name}", "-")
        prauc_value = self.metrics.get(f"prauc_{score_name}", "-")

        self.info_threshold.setText(
            f"Method={method} | Dùng {score_name}: "
            f"score={score_used:.6f} vs threshold={thr_used:.6f}"
        )

        self.info_direction.setText(
            f"Direction: higher_is_anomaly | "
            f"AUROC {score_name}={self.fmt_float(auroc_value)} | "
            f"PR-AUC {score_name}={self.fmt_float(prauc_value)}"
        )

        anomaly = score_used > thr_used

        if anomaly:
            self.result.setText("BẤT THƯỜNG (ANOMALY)")
            self.result.setStyleSheet(
                "border:3px solid red; color:red; font-size:20px; "
                "font-weight:bold; background:#fff0f0;"
            )
        else:
            self.result.setText("BÌNH THƯỜNG (NORMAL)")
            self.result.setStyleSheet(
                "border:3px solid green; color:green; font-size:20px; "
                "font-weight:bold; background:#f0fff0;"
            )

    # =========================
    # CHART
    # =========================
    def load_chart(self):
        name = self.combo_chart.currentText()
        path = os.path.join(self.exp_dir(), f"{name}.png")

        if os.path.exists(path):
            self.set_fit_pixmap(self.chart, QtGui.QPixmap(path))
        else:
            self.chart.clear()
            self.chart.setText(f"Không có biểu đồ:\n{path}")

    # =========================
    # HISTORY
    # =========================
    def load_history(self):
        log_path = os.path.join(self.hist_exp_dir(), "train_log.json")
        self.history_table.setRowCount(0)
        self.history_summary.clear()

        if not os.path.exists(log_path):
            self.history_summary.setText(
                f"Chưa có lịch sử train.\n"
                f"Tìm tại: {log_path}\n\n"
                f"Hãy chạy doan.py trước."
            )
            return

        try:
            with open(log_path, "r", encoding="utf-8") as f:
                h = json.load(f)
        except Exception as e:
            self.history_summary.setText(f"Lỗi đọc file: {e}")
            return

        logs = h.get("epoch_logs", [])
        self.history_table.setRowCount(len(logs))

        for i, e in enumerate(logs):
            self.history_table.setItem(
                i, 0, QtWidgets.QTableWidgetItem(str(e.get("epoch", "")))
            )
            self.history_table.setItem(
                i, 1, QtWidgets.QTableWidgetItem(self.fmt_float(e.get("train_loss"), 6))
            )
            self.history_table.setItem(
                i, 2, QtWidgets.QTableWidgetItem(self.fmt_float(e.get("val_loss"), 6))
            )
            self.history_table.setItem(
                i, 3, QtWidgets.QTableWidgetItem(f"{float(e.get('lr', 0)):.2e}")
            )
            self.history_table.setItem(
                i, 4, QtWidgets.QTableWidgetItem(e.get("timestamp", ""))
            )

        self.history_table.resizeColumnsToContents()

        if logs:
            self.history_table.scrollToBottom()

        tl = h.get("train_losses", [])
        vl = h.get("val_losses", [])

        lines = [
            "═══ THÔNG TIN THÍ NGHIỆM ═══",
            "",
            f"Tên           : {h.get('exp_name', '-')}",
            f"Dataset       : {h.get('dataset', '-')}",
            f"Model         : {h.get('model_name', '-')}",
            f"Optimizer     : {h.get('optimizer', '-')}",
            f"Regularization: {h.get('regularization', '-')}",
            f"Augmentation  : {h.get('augmentation', '-')}",
            f"Số tham số    : {h.get('params', '-'):,}" if isinstance(h.get("params"), int) else f"Số tham số    : {h.get('params', '-')}",
            "",
            f"Epochs        : {h.get('epochs', '-')}",
            f"Batch size    : {h.get('batch_size', '-')}",
            f"LR ban đầu    : {h.get('lr', '-')}",
            f"Device        : {h.get('device', '-')}",
            "",
            f"Bắt đầu       : {h.get('started_at', '-')}",
            f"Kết thúc      : {h.get('finished_at', 'Đang train...')}",
            "",
            f"Tiến trình    : {len(logs)}/{h.get('epochs', 0)} epochs",
        ]

        if tl:
            lines.append(f"Train loss    : {tl[0]:.6f} → {tl[-1]:.6f}")

        if vl:
            lines.append(f"Val loss      : {vl[0]:.6f} → {vl[-1]:.6f}")
            best_ep = int(np.argmin(vl)) + 1
            lines.append(f"Best val ep   : epoch {best_ep} ({min(vl):.6f})")

        m = h.get("metrics")

        if m:
            lines += [
                "",
                "═══ KẾT QUẢ ĐÁNH GIÁ ═══",
                "",
                f"AUROC pixel     : {self.fmt_float(m.get('auroc_pixel'))}",
                f"PR-AUC pixel    : {self.fmt_float(m.get('prauc_pixel'))}",
                f"AUROC feature   : {self.fmt_float(m.get('auroc_feature'))}",
                f"PR-AUC feature  : {self.fmt_float(m.get('prauc_feature'))}",
                f"AUROC combined  : {self.fmt_float(m.get('auroc_combined'))}",
                f"PR-AUC combined : {self.fmt_float(m.get('prauc_combined'))}",
                "",
                f"Best F1 pixel    : {self.fmt_float(m.get('best_f1_pixel'))}",
                f"Best F1 feature  : {self.fmt_float(m.get('best_f1_feature'))}",
                f"Best F1 combined : {self.fmt_float(m.get('best_f1_combined'))}",
                "",
                f"Threshold method   : {m.get('selected_threshold_method', '-')}",
                f"Selected score     : {m.get('selected_score', '-')}",
                f"Threshold pixel    : {self.fmt_float(m.get('threshold_pixel'), 6)}",
                f"Threshold feature  : {self.fmt_float(m.get('threshold_feature'), 6)}",
                f"Threshold combined : {self.fmt_float(m.get('threshold_combined'), 6)}",
            ]
        else:
            lines.append("\n(Chưa có metrics – train chưa xong)")

        self.history_summary.setText("\n".join(lines))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = QtWidgets.QMainWindow()

    ui = UI()
    ui.setup(window)

    window.show()
    sys.exit(app.exec())
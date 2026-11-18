import sys
import os
import csv
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QLineEdit, QPushButton, QSlider, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox, QProgressBar,
    QFrame, QMessageBox, QFileDialog, QComboBox, QScrollArea, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor

# Matplotlib integration for PyQt5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from monte_carlo import SimulationConfig, run_monte_carlo, TargetHitStats
from black76 import black76_price

DARK_STYLE = """
QMainWindow, QWidget {
    background-color: #121418;
    color: #E2E8F0;
    font-family: 'Segoe UI', -apple-system, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #2D3748;
    border-radius: 8px;
    margin-top: 10px;
    font-weight: bold;
    color: #63B3ED;
    padding-top: 12px;
    background-color: #1A202C;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    background-color: #1A202C;
}
QLabel {
    color: #CBD5E0;
    font-size: 13px;
}
/* Enhanced font size and JetBrains Mono for all inputs */
QLineEdit, QComboBox {
    background-color: #0D1117;
    border: 1.5px solid #4A5568;
    border-radius: 5px;
    padding: 4px 10px;
    color: #F7FAFC;
    font-family: 'JetBrains Mono', 'Consolas', 'Cascadia Mono', monospace;
    font-size: 14px;
    font-weight: 500;
    min-height: 28px;
    max-height: 32px;
}
QLineEdit:focus, QComboBox:focus {
    border: 1.5px solid #3182CE;
    background-color: #161B22;
    color: #FFFFFF;
}
QPushButton {
    background-color: #2B6CB0;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: bold;
    font-size: 13px;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #3182CE;
}
QPushButton:pressed {
    background-color: #2C5282;
}
QPushButton#runBtn {
    background-color: #2F855A;
    font-size: 14px;
    font-weight: 700;
    padding: 10px 18px;
    min-height: 36px;
    border-radius: 6px;
}
QPushButton#runBtn:hover {
    background-color: #38A169;
}
QPushButton#atmBtn {
    background-color: #2D3748;
    border: 1px solid #4A5568;
    color: #E2E8F0;
    font-size: 12px;
    padding: 4px 8px;
    min-height: 26px;
}
QPushButton#atmBtn:hover {
    background-color: #4A5568;
    color: #FFFFFF;
}
QPushButton#exportBtn {
    background-color: #4A5568;
    min-height: 26px;
}
QPushButton#exportBtn:hover {
    background-color: #718096;
}
QSlider::groove:horizontal {
    border: 1px solid #4A5568;
    height: 6px;
    background: #2D3748;
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: #3182CE;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #E2E8F0;
    border: 1px solid #CBD5E0;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #FFFFFF;
}
QTabWidget::pane {
    border: 1px solid #2D3748;
    background-color: #1A202C;
    border-radius: 6px;
}
QTabBar::tab {
    background: #171923;
    color: #A0AEC0;
    padding: 7px 14px;
    margin-right: 3px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    border: 1px solid #2D3748;
    border-bottom: none;
    font-weight: 600;
    font-size: 12px;
}
QTabBar::tab:selected {
    background: #1A202C;
    color: #63B3ED;
    border-bottom: 2px solid #3182CE;
}
QTabBar::tab:hover:!selected {
    background: #2D3748;
    color: #E2E8F0;
}
QTableWidget {
    background-color: #0D1117;
    gridline-color: #2D3748;
    border: 1px solid #2D3748;
    color: #E2E8F0;
    border-radius: 6px;
    font-family: 'JetBrains Mono', 'Consolas', 'Cascadia Mono', monospace;
    font-size: 12px;
}
QHeaderView::section {
    background-color: #1A202C;
    color: #A0AEC0;
    font-weight: bold;
    padding: 5px;
    border: 1px solid #2D3748;
    font-family: 'Segoe UI', sans-serif;
    font-size: 12px;
}
QTableWidget::item:selected {
    background-color: #2B6CB0;
    color: white;
}
QProgressBar {
    border: 1px solid #4A5568;
    border-radius: 4px;
    text-align: center;
    background-color: #0D1117;
    color: #E2E8F0;
    height: 16px;
    font-size: 11px;
}
QProgressBar::chunk {
    background-color: #3182CE;
    border-radius: 3px;
}
QScrollBar:vertical {
    border: none;
    background: #1A202C;
    width: 8px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #4A5568;
    min-height: 20px;
    border-radius: 4px;
}
QScrollBar::handle:vertical:hover {
    background: #718096;
}
"""

SLIDER_SCALE = 1000.0  # Scale factor: 1 unit on slider = 0.001 beta


class SimulationWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, cfg: SimulationConfig):
        super().__init__()
        self.cfg = cfg

    def run(self):
        try:
            res = run_monte_carlo(self.cfg)
            self.finished.emit(res)
        except Exception as e:
            self.error.emit(str(e))


class MetricCard(QFrame):
    def __init__(self, title: str, subtitle: str = "", accent_color: str = "#3182CE"):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: #171923;
                border: 1.5px solid #2D3748;
                border-left: 5px solid {accent_color};
                border-radius: 8px;
                padding: 10px 12px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        # Card title (e.g. CALL AVG (EXPECTED))
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("color: #A0AEC0; font-size: 11px; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;")
        layout.addWidget(self.title_label)

        # Big primary value (e.g. ₹ 135.45)
        self.value_label = QLabel("--")
        self.value_label.setStyleSheet("color: #FFFFFF; font-size: 28px; font-weight: 800; font-family: 'JetBrains Mono', 'Consolas', monospace;")
        layout.addWidget(self.value_label)

        # Secondary label - now large enough to actually read
        self.sub_label = QLabel(subtitle)
        self.sub_label.setStyleSheet("color: #E2E8F0; font-size: 14px; font-weight: 600; font-family: 'JetBrains Mono', 'Consolas', monospace;")
        layout.addWidget(self.sub_label)

    def set_data(self, value: str, sub: str = ""):
        self.value_label.setText(value)
        if sub:
            self.sub_label.setText(sub)


class OptionsSimApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OptionSims - Developed by Anupam Dutta")
        self.resize(1400, 920)
        self.setMinimumSize(1100, 720)
        self.setStyleSheet(DARK_STYLE)

        self.sim_results = None
        self.worker = None

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        root_layout = QVBoxLayout(main_widget)
        root_layout.setContentsMargins(12, 8, 12, 8)
        root_layout.setSpacing(8)

        # 1. Top Header: App Name & Subtitle (Compact)
        header_widget = self.create_header()
        root_layout.addWidget(header_widget)

        # 2. Main Content (Left: Scrollable Inputs, Right: Tabs)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)

        # Left Scrollable Input Area
        scroll_inputs = QScrollArea()
        scroll_inputs.setWidgetResizable(True)
        scroll_inputs.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_inputs.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_inputs.setStyleSheet("""
            QScrollArea {
                border: 1px solid #2D3748;
                border-radius: 8px;
                background-color: #161B22;
            }
        """)

        left_panel = self.create_input_panel()
        scroll_inputs.setWidget(left_panel)
        content_layout.addWidget(scroll_inputs, stretch=3)

        # Right Panel: Results & Visualizations
        right_panel = self.create_results_panel()
        content_layout.addWidget(right_panel, stretch=7)

        root_layout.addLayout(content_layout, stretch=1)

        # 3. Footer Bar: GammaGrid Systems
        footer_widget = self.create_footer()
        root_layout.addWidget(footer_widget)

        # Initialize default preview calculations
        self.on_param_changed()

    def create_header(self) -> QWidget:
        header = QFrame()
        header.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1A202C, stop:1 #171923);
                border: 1px solid #2D3748;
                border-radius: 6px;
                padding: 4px 12px;
            }
        """)
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 2, 8, 2)

        # Left: App Name + Subtitle inline
        title_lbl = QLabel("OptionSims")
        title_lbl.setStyleSheet("""
            font-size: 20px;
            font-weight: 900;
            color: #63B3ED;
            letter-spacing: 0.5px;
            font-family: 'Segoe UI', sans-serif;
        """)
        h_layout.addWidget(title_lbl)

        sep = QLabel("|")
        sep.setStyleSheet("color: #4A5568; font-size: 16px; margin: 0 4px;")
        h_layout.addWidget(sep)

        sub_lbl = QLabel("Developed by Anupam Dutta")
        sub_lbl.setStyleSheet("""
            font-size: 12px;
            font-weight: 600;
            color: #A0AEC0;
        """)
        h_layout.addWidget(sub_lbl)

        h_layout.addStretch()

        badge = QLabel("⚡ Black-76 Derivatives Simulation Engine")
        badge.setStyleSheet("""
            background-color: #2B6CB0;
            color: #FFFFFF;
            font-weight: bold;
            font-size: 11px;
            padding: 4px 10px;
            border-radius: 10px;
        """)
        h_layout.addWidget(badge)

        return header

    def create_footer(self) -> QWidget:
        footer = QFrame()
        footer.setStyleSheet("""
            QFrame {
                background-color: #171923;
                border: 1px solid #2D3748;
                border-radius: 5px;
                padding: 2px 10px;
            }
        """)
        f_layout = QHBoxLayout(footer)
        f_layout.setContentsMargins(6, 2, 6, 2)

        lbl_footer_brand = QLabel("⚡ GammaGrid Systems")
        lbl_footer_brand.setStyleSheet("""
            font-weight: 800;
            color: #4FD1C5;
            font-size: 12px;
            letter-spacing: 0.5px;
        """)
        f_layout.addWidget(lbl_footer_brand)

        f_layout.addStretch()

        lbl_desc = QLabel("Quantitative Analytics • 9:15 AM - 3:40 PM Session (385m) • 365-Day Annualization")
        lbl_desc.setStyleSheet("color: #718096; font-size: 11px;")
        f_layout.addWidget(lbl_desc)

        return footer

    def create_input_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.setSpacing(10)

        # Group 1: Underlying & Targets
        grp_sf = QGroupBox("Synthetic Future & Targets")
        grid_sf = QGridLayout(grp_sf)
        grid_sf.setContentsMargins(10, 12, 10, 10)
        grid_sf.setVerticalSpacing(8)
        grid_sf.setHorizontalSpacing(8)

        grid_sf.addWidget(QLabel("Opening SF (F₀):"), 0, 0)
        self.txt_sf = QLineEdit("8296.3")
        grid_sf.addWidget(self.txt_sf, 0, 1)

        grid_sf.addWidget(QLabel("Projected Upside:"), 1, 0)
        self.txt_upside = QLineEdit("8350.0")
        grid_sf.addWidget(self.txt_upside, 1, 1)

        grid_sf.addWidget(QLabel("Projected Downside:"), 2, 0)
        self.txt_downside = QLineEdit("8185.8")
        grid_sf.addWidget(self.txt_downside, 2, 1)

        layout.addWidget(grp_sf)

        # Group 2: Option Strikes & Expiry
        grp_opt = QGroupBox("Option Parameters & Expiry")
        grid_opt = QGridLayout(grp_opt)
        grid_opt.setContentsMargins(10, 12, 10, 10)
        grid_opt.setVerticalSpacing(8)
        grid_opt.setHorizontalSpacing(8)

        grid_opt.addWidget(QLabel("Call Strike (K_call):"), 0, 0)
        self.txt_call_k = QLineEdit("8300.0")
        grid_opt.addWidget(self.txt_call_k, 0, 1)

        grid_opt.addWidget(QLabel("Put Strike (K_put):"), 1, 0)
        self.txt_put_k = QLineEdit("8300.0")
        grid_opt.addWidget(self.txt_put_k, 1, 1)

        btn_atm = QPushButton("Set ATM to SF")
        btn_atm.setObjectName("atmBtn")
        btn_atm.clicked.connect(self.set_strikes_atm)
        grid_opt.addWidget(btn_atm, 2, 0, 1, 2)

        grid_opt.addWidget(QLabel("DTE (Days):"), 3, 0)
        self.txt_dte = QLineEdit("1.621528")
        grid_opt.addWidget(self.txt_dte, 3, 1)

        grid_opt.addWidget(QLabel("Opening IV (%):"), 4, 0)
        self.txt_iv = QLineEdit("54.043")
        grid_opt.addWidget(self.txt_iv, 4, 1)

        grid_opt.addWidget(QLabel("Risk-free rate r (%):"), 5, 0)
        self.txt_r = QLineEdit("0.0")
        grid_opt.addWidget(self.txt_r, 5, 1)

        layout.addWidget(grp_opt)

        # Group 3: Market Hours info
        grp_timing = QGroupBox("Trading Session & Model")
        v_timing = QVBoxLayout(grp_timing)
        v_timing.setContentsMargins(10, 12, 10, 10)
        lbl_timing = QLabel("• Market Hours: 9:15 AM - 3:40 PM (385 mins/day)\n"
                            "• Black-76 Framework: Drift = 0 on SF\n"
                            "• Expiry Time Basis: 365 Days Annualization")
        lbl_timing.setStyleSheet("color: #A0AEC0; font-size: 11px; line-height: 1.4;")
        v_timing.addWidget(lbl_timing)
        layout.addWidget(grp_timing)

        # Group 4: IV Skew Sensitivity Slider & Input Box
        grp_skew = QGroupBox("IV Skew Sensitivity (β)")
        v_skew = QVBoxLayout(grp_skew)
        v_skew.setContentsMargins(10, 12, 10, 10)
        v_skew.setSpacing(6)

        # Row: Sensitivity factor label + QLineEdit + Skew type badge
        h_skew_input = QHBoxLayout()
        h_skew_input.setSpacing(6)
        lbl_skew_title = QLabel("Sensitivity Factor (β):")
        h_skew_input.addWidget(lbl_skew_title)

        self.txt_skew = QLineEdit("-0.08")
        self.txt_skew.setFixedWidth(85)
        self.txt_skew.setToolTip("Enter skew sensitivity β (e.g. -0.08)")
        h_skew_input.addWidget(self.txt_skew)

        self.lbl_skew_type = QLabel("Negative Skew")
        self.lbl_skew_type.setStyleSheet("color: #4FD1C5; font-weight: bold; font-size: 11px;")
        h_skew_input.addWidget(self.lbl_skew_type)
        h_skew_input.addStretch()
        v_skew.addLayout(h_skew_input)

        # Horizontal Slider
        self.slider_skew = QSlider(Qt.Horizontal)
        self.slider_skew.setRange(-2000, 2000)
        self.slider_skew.setValue(int(round(-0.08 * SLIDER_SCALE)))
        v_skew.addWidget(self.slider_skew)

        # Live IV preview card
        self.card_iv_preview = QFrame()
        self.card_iv_preview.setStyleSheet("""
            QFrame {
                background-color: #0D1117;
                border: 1px solid #2D3748;
                border-radius: 5px;
                padding: 4px 8px;
            }
        """)
        v_preview = QVBoxLayout(self.card_iv_preview)
        v_preview.setContentsMargins(6, 4, 6, 4)
        v_preview.setSpacing(2)

        self.lbl_iv_preview = QLabel("Est. IV at Upside: 53.99%  |  Downside: 54.15%")
        self.lbl_iv_preview.setStyleSheet("color: #E2E8F0; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        v_preview.addWidget(self.lbl_iv_preview)

        self.lbl_iv_deltas = QLabel("ΔIV Upside: -0.05%  |  ΔIV Downside: +0.11%")
        self.lbl_iv_deltas.setStyleSheet("color: #A0AEC0; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 10px;")
        v_preview.addWidget(self.lbl_iv_deltas)

        v_skew.addWidget(self.card_iv_preview)
        layout.addWidget(grp_skew)

        # Group 5: Simulation Settings
        grp_sim = QGroupBox("Monte Carlo Settings")
        grid_sim = QGridLayout(grp_sim)
        grid_sim.setContentsMargins(10, 12, 10, 10)
        grid_sim.setVerticalSpacing(8)
        grid_sim.setHorizontalSpacing(8)

        grid_sim.addWidget(QLabel("Paths (N):"), 0, 0)
        self.combo_paths = QComboBox()
        self.combo_paths.addItems(["5000", "10000", "20000", "30000"])
        self.combo_paths.setCurrentText("10000")
        grid_sim.addWidget(self.combo_paths, 0, 1)

        grid_sim.addWidget(QLabel("Time Steps:"), 1, 0)
        self.combo_steps = QComboBox()
        self.combo_steps.addItems(["200", "385 (1-min/day)", "600", "1000"])
        self.combo_steps.setCurrentText("385 (1-min/day)")
        grid_sim.addWidget(self.combo_steps, 1, 1)

        layout.addWidget(grp_sim)

        # Action Buttons
        self.btn_run = QPushButton("▶ Run Monte Carlo Simulation")
        self.btn_run.setObjectName("runBtn")
        self.btn_run.clicked.connect(self.run_simulation)
        layout.addWidget(self.btn_run)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Baseline Display
        self.lbl_baseline = QLabel("Baseline Black-76: Call = -- | Put = --")
        self.lbl_baseline.setAlignment(Qt.AlignCenter)
        self.lbl_baseline.setStyleSheet("color: #E2E8F0; font-weight: bold; font-family: 'JetBrains Mono', 'Consolas', monospace; font-size: 12px; padding: 6px; background: #171923; border: 1px solid #2D3748; border-radius: 4px;")
        layout.addWidget(self.lbl_baseline)

        # Connect signals for interactive real-time updates
        self.slider_skew.valueChanged.connect(self.on_slider_moved)
        self.txt_skew.textChanged.connect(self.on_skew_text_edited)

        # Connect parameter changes to live updates
        self.txt_sf.textChanged.connect(self.on_param_changed)
        self.txt_upside.textChanged.connect(self.on_param_changed)
        self.txt_downside.textChanged.connect(self.on_param_changed)
        self.txt_iv.textChanged.connect(self.on_param_changed)
        self.txt_dte.textChanged.connect(self.on_param_changed)
        self.txt_call_k.textChanged.connect(self.on_param_changed)
        self.txt_put_k.textChanged.connect(self.on_param_changed)
        self.txt_r.textChanged.connect(self.on_param_changed)

        layout.addStretch()
        return container

    def create_results_panel(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.tabs = QTabWidget()

        # Tab 1: Executive Summary & Scenarios
        tab_summary = self.create_summary_tab()
        self.tabs.addTab(tab_summary, "📊 Scenario Summary")

        # Tab 2: SF Monte Carlo Trajectory Paths
        tab_paths = self.create_paths_tab()
        self.tabs.addTab(tab_paths, "📈 SF Price Paths")

        # Tab 3: Price Distribution Histograms
        tab_dist = self.create_distribution_tab()
        self.tabs.addTab(tab_dist, "📉 Price Distributions")

        # Tab 4: All Simulations Inspector Table
        tab_table = self.create_table_tab()
        self.tabs.addTab(tab_table, "📋 All Simulations Data")

        layout.addWidget(self.tabs)
        return container

    def create_summary_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(12)

        # Upside Target Section
        grp_up = QGroupBox("🔺 SCENARIO 1: SF HITS PROJECTED UPSIDE TARGET")
        layout_up = QVBoxLayout(grp_up)
        layout_up.setContentsMargins(12, 14, 12, 12)
        layout_up.setSpacing(8)

        self.lbl_up_meta = QLabel("Target: -- | Hit Probability: -- | Avg Time to Hit: -- | Est. IV at Target: --")
        self.lbl_up_meta.setStyleSheet("""
            color: #68D391;
            font-weight: bold;
            font-size: 13.5px;
            padding: 8px 12px;
            background-color: #0D1117;
            border: 1px solid #276749;
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        """)
        layout_up.addWidget(self.lbl_up_meta)

        grid_up = QGridLayout()
        grid_up.setSpacing(8)

        self.card_up_call_mean = MetricCard("Call Avg (Expected)", "Expected Call Price", "#38A169")
        self.card_up_call_best = MetricCard("Call Best Case", "Max Simulated Price", "#48BB78")
        self.card_up_call_range = MetricCard("Call 5th - 95th %ile", "Realistic Range", "#2F855A")

        self.card_up_put_mean = MetricCard("Put Avg (Expected)", "Expected Put Price", "#E53E3E")
        self.card_up_put_best = MetricCard("Put Best Case", "Max Simulated Price", "#FC8181")
        self.card_up_put_range = MetricCard("Put 5th - 95th %ile", "Realistic Range", "#9B2C2C")

        grid_up.addWidget(self.card_up_call_mean, 0, 0)
        grid_up.addWidget(self.card_up_call_best, 0, 1)
        grid_up.addWidget(self.card_up_call_range, 0, 2)
        grid_up.addWidget(self.card_up_put_mean, 1, 0)
        grid_up.addWidget(self.card_up_put_best, 1, 1)
        grid_up.addWidget(self.card_up_put_range, 1, 2)
        layout_up.addLayout(grid_up)
        layout.addWidget(grp_up)

        # Downside Target Section
        grp_down = QGroupBox("🔻 SCENARIO 2: SF HITS PROJECTED DOWNSIDE TARGET")
        layout_down = QVBoxLayout(grp_down)
        layout_down.setContentsMargins(12, 14, 12, 12)
        layout_down.setSpacing(8)

        self.lbl_down_meta = QLabel("Target: -- | Hit Probability: -- | Avg Time to Hit: -- | Est. IV at Target: --")
        self.lbl_down_meta.setStyleSheet("""
            color: #FC8181;
            font-weight: bold;
            font-size: 13.5px;
            padding: 8px 12px;
            background-color: #0D1117;
            border: 1px solid #9B2C2C;
            border-radius: 6px;
            font-family: 'JetBrains Mono', 'Consolas', monospace;
        """)
        layout_down.addWidget(self.lbl_down_meta)

        grid_down = QGridLayout()
        grid_down.setSpacing(8)

        self.card_down_call_mean = MetricCard("Call Avg (Expected)", "Expected Call Price", "#38A169")
        self.card_down_call_best = MetricCard("Call Best Case", "Max Simulated Price", "#48BB78")
        self.card_down_call_range = MetricCard("Call 5th - 95th %ile", "Realistic Range", "#2F855A")

        self.card_down_put_mean = MetricCard("Put Avg (Expected)", "Expected Put Price", "#E53E3E")
        self.card_down_put_best = MetricCard("Put Best Case", "Max Simulated Price", "#FC8181")
        self.card_down_put_range = MetricCard("Put 5th - 95th %ile", "Realistic Range", "#9B2C2C")

        grid_down.addWidget(self.card_down_call_mean, 0, 0)
        grid_down.addWidget(self.card_down_call_best, 0, 1)
        grid_down.addWidget(self.card_down_call_range, 0, 2)
        grid_down.addWidget(self.card_down_put_mean, 1, 0)
        grid_down.addWidget(self.card_down_put_best, 1, 1)
        grid_down.addWidget(self.card_down_put_range, 1, 2)
        layout_down.addLayout(grid_down)
        layout.addWidget(grp_down)

        layout.addStretch()
        return widget

    def create_paths_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        self.fig_paths = Figure(facecolor='#121418', edgecolor='none')
        self.canvas_paths = FigureCanvas(self.fig_paths)
        layout.addWidget(self.canvas_paths)
        return widget

    def create_distribution_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Informational Axis Explanation Banner
        info_banner = QFrame()
        info_banner.setStyleSheet("""
            QFrame {
                background-color: #171923;
                border: 1.5px solid #2D3748;
                border-left: 5px solid #3182CE;
                border-radius: 6px;
                padding: 6px 12px;
            }
        """)
        v_info = QVBoxLayout(info_banner)
        v_info.setContentsMargins(8, 6, 8, 6)
        v_info.setSpacing(3)

        lbl_guide = QLabel("📊  X-Axis = Estimated Option Price (₹)   |   Y-Axis = Path Count (Simulation Frequency)")
        lbl_guide.setStyleSheet("color: #63B3ED; font-weight: 800; font-size: 14px; font-family: 'Segoe UI', sans-serif;")
        v_info.addWidget(lbl_guide)

        lbl_guide_sub = QLabel(
            "X-Axis: Black-76 option price in ₹ at the exact moment SF hits the target.   "
            "Y-Axis: Number of simulated paths that produced that price."
        )
        lbl_guide_sub.setStyleSheet("color: #A0AEC0; font-size: 12px;")
        v_info.addWidget(lbl_guide_sub)

        layout.addWidget(info_banner)

        self.fig_dist = Figure(facecolor='#121418', edgecolor='none')
        self.canvas_dist = FigureCanvas(self.fig_dist)
        layout.addWidget(self.canvas_dist)
        return widget

    def create_table_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Controls over table
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("Filter Scenario:"))
        self.combo_table_filter = QComboBox()
        self.combo_table_filter.addItems(["All Hitting Simulations", "Upside Target Hits Only", "Downside Target Hits Only"])
        self.combo_table_filter.currentIndexChanged.connect(self.populate_table)
        ctrl_layout.addWidget(self.combo_table_filter)

        ctrl_layout.addStretch()

        self.btn_export = QPushButton("💾 Export to CSV")
        self.btn_export.setObjectName("exportBtn")
        self.btn_export.clicked.connect(self.export_csv)
        ctrl_layout.addWidget(self.btn_export)
        layout.addLayout(ctrl_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Path ID", "Event / Target Hit", "SF Level", "Time Elapsed (Mins)",
            "Elapsed DTE", "Remaining DTE", "Call Price", "Put Price"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setStyleSheet("alternate-background-color: #161B22;")
        layout.addWidget(self.table)

        self.lbl_table_count = QLabel("Displaying 0 simulations")
        self.lbl_table_count.setStyleSheet("color: #718096; font-size: 11px;")
        layout.addWidget(self.lbl_table_count)

        return widget

    def set_strikes_atm(self):
        try:
            sf_val = float(self.txt_sf.text().strip())
            self.txt_call_k.setText(f"{sf_val:.1f}")
            self.txt_put_k.setText(f"{sf_val:.1f}")
            self.on_param_changed()
        except ValueError:
            pass

    def on_slider_moved(self):
        val = self.slider_skew.value()
        beta = val / SLIDER_SCALE

        self.txt_skew.blockSignals(True)
        self.txt_skew.setText(f"{beta:.4f}".rstrip('0').rstrip('.'))
        self.txt_skew.blockSignals(False)

        self.update_skew_labels(beta)
        self.update_iv_preview(beta)

    def on_skew_text_edited(self, text: str):
        try:
            clean_txt = text.strip()
            if not clean_txt or clean_txt in ('-', '+', '.'):
                return
            beta = float(clean_txt)
            slider_val = int(round(beta * SLIDER_SCALE))
            slider_val = max(-2000, min(2000, slider_val))

            self.slider_skew.blockSignals(True)
            self.slider_skew.setValue(slider_val)
            self.slider_skew.blockSignals(False)

            self.update_skew_labels(beta)
            self.update_iv_preview(beta)
        except ValueError:
            pass

    def update_skew_labels(self, beta: float):
        if beta < -0.001:
            skew_type = "Negative Skew"
            col = "#4FD1C5"
        elif beta > 0.001:
            skew_type = "Positive Skew"
            col = "#F6AD55"
        else:
            skew_type = "Zero Skew (Flat)"
            col = "#A0AEC0"

        self.lbl_skew_type.setText(skew_type)
        self.lbl_skew_type.setStyleSheet(f"color: {col}; font-weight: bold; font-size: 11px;")

    def update_iv_preview(self, beta: float = None):
        try:
            if beta is None:
                try:
                    beta = float(self.txt_skew.text().strip())
                except ValueError:
                    beta = self.slider_skew.value() / SLIDER_SCALE

            sf = float(self.txt_sf.text().strip())
            up = float(self.txt_upside.text().strip())
            down = float(self.txt_downside.text().strip())
            base_iv = float(self.txt_iv.text().strip()) / 100.0

            d_up = (up - sf) / sf
            d_down = (down - sf) / sf

            delta_iv_up = beta * d_up
            delta_iv_down = beta * d_down

            iv_up = max(0.01, base_iv + delta_iv_up) * 100.0
            iv_down = max(0.01, base_iv + delta_iv_down) * 100.0

            self.lbl_iv_preview.setText(
                f"Est. IV at Upside: {iv_up:.2f}%  |  Downside: {iv_down:.2f}%"
            )
            self.lbl_iv_deltas.setText(
                f"ΔIV at Upside: {(delta_iv_up * 100.0):+.2f}%  |  ΔIV at Downside: {(delta_iv_down * 100.0):+.2f}%"
            )
        except Exception:
            self.lbl_iv_preview.setText("Est. IV at Targets: Check Inputs")
            self.lbl_iv_deltas.setText("")

    def on_param_changed(self):
        try:
            sf = float(self.txt_sf.text().strip())
            call_k = float(self.txt_call_k.text().strip())
            put_k = float(self.txt_put_k.text().strip())
            dte = float(self.txt_dte.text().strip())
            iv = float(self.txt_iv.text().strip()) / 100.0
            r = float(self.txt_r.text().strip()) / 100.0

            T = dte / 365.0
            base_call = black76_price(sf, call_k, T, r, iv, 'call')
            base_put = black76_price(sf, put_k, T, r, iv, 'put')
            self.lbl_baseline.setText(f"Baseline Black-76 (t=0): Call = ₹{base_call:.2f} | Put = ₹{base_put:.2f}")

            self.update_iv_preview()
        except Exception:
            self.lbl_baseline.setText("Baseline Black-76: --")

    def run_simulation(self):
        try:
            sf = float(self.txt_sf.text().strip())
            up = float(self.txt_upside.text().strip())
            down = float(self.txt_downside.text().strip())
            call_k = float(self.txt_call_k.text().strip())
            put_k = float(self.txt_put_k.text().strip())
            dte = float(self.txt_dte.text().strip())
            iv = float(self.txt_iv.text().strip()) / 100.0
            r = float(self.txt_r.text().strip()) / 100.0
            paths = int(self.combo_paths.currentText())
            steps_txt = self.combo_steps.currentText().split()[0]
            steps = int(steps_txt)

            try:
                beta = float(self.txt_skew.text().strip())
            except ValueError:
                beta = self.slider_skew.value() / SLIDER_SCALE

            if up < down:
                QMessageBox.warning(self, "Target Level Warning", "Upside target must be greater than Downside target. Swapping targets automatically.")
                up, down = down, up
                self.txt_upside.setText(str(up))
                self.txt_downside.setText(str(down))

            cfg = SimulationConfig(
                opening_sf=sf,
                upside_target=up,
                downside_target=down,
                dte_days=dte,
                opening_iv=iv,
                r=r,
                call_strike=call_k,
                put_strike=put_k,
                num_simulations=paths,
                time_steps=steps,
                iv_skew_beta=beta
            )

            self.btn_run.setEnabled(False)
            self.btn_run.setText("⏳ Simulating...")
            self.progress_bar.setVisible(True)

            self.worker = SimulationWorker(cfg)
            self.worker.finished.connect(self.on_sim_finished)
            self.worker.error.connect(self.on_sim_error)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Invalid Inputs", f"Please check all input fields:\n{str(e)}")

    def on_sim_error(self, err_msg: str):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ Run Monte Carlo Simulation")
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Simulation Error", f"An error occurred during simulation:\n{err_msg}")

    def on_sim_finished(self, results: dict):
        self.btn_run.setEnabled(True)
        self.btn_run.setText("▶ Run Monte Carlo Simulation")
        self.progress_bar.setVisible(False)
        self.sim_results = results

        self.lbl_baseline.setText(f"Baseline Black-76: Call = ₹{results['opening_call']:.2f} | Put = ₹{results['opening_put']:.2f}")
        self.update_summary_cards(results)
        self.plot_paths(results)
        self.plot_distributions(results)
        self.populate_table()

    def update_summary_cards(self, res: dict):
        up: TargetHitStats = res['upside_stats']
        down: TargetHitStats = res['downside_stats']

        def fmt_time(minutes: float) -> str:
            hours = minutes / 60.0
            return f"{minutes:.1f}m ({hours:.2f}h)"

        # Upside cards
        self.lbl_up_meta.setText(
            f"🎯 Target Level: {up.target_level:.1f}  |  Hit Probability: {up.hit_probability_pct:.1f}% ({up.hit_count}/{up.total_paths})  |  "
            f"Avg Time to Hit: {fmt_time(up.avg_elapsed_minutes)}  |  Model IV at Target: {up.iv_at_hit_pct:.2f}%"
        )
        if up.hit_count > 0:
            self.card_up_call_mean.set_data(f"₹ {up.call_mean:.2f}", f"Median: ₹ {up.call_median:.2f}")
            self.card_up_call_best.set_data(f"₹ {up.call_max:.2f}", f"Worst Case: ₹ {up.call_min:.2f}")
            self.card_up_call_range.set_data(f"₹ {up.call_p05:.2f} - ₹ {up.call_p95:.2f}", f"Std Dev: ₹ {up.call_std:.2f}")

            self.card_up_put_mean.set_data(f"₹ {up.put_mean:.2f}", f"Median: ₹ {up.put_median:.2f}")
            self.card_up_put_best.set_data(f"₹ {up.put_max:.2f}", f"Worst Case: ₹ {up.put_min:.2f}")
            self.card_up_put_range.set_data(f"₹ {up.put_p05:.2f} - ₹ {up.put_p95:.2f}", f"Std Dev: ₹ {up.put_std:.2f}")
        else:
            for card in [self.card_up_call_mean, self.card_up_call_best, self.card_up_call_range,
                         self.card_up_put_mean, self.card_up_put_best, self.card_up_put_range]:
                card.set_data("No Hits", "Target not reached")

        # Downside cards
        self.lbl_down_meta.setText(
            f"🎯 Target Level: {down.target_level:.1f}  |  Hit Probability: {down.hit_probability_pct:.1f}% ({down.hit_count}/{down.total_paths})  |  "
            f"Avg Time to Hit: {fmt_time(down.avg_elapsed_minutes)}  |  Model IV at Target: {down.iv_at_hit_pct:.2f}%"
        )
        if down.hit_count > 0:
            self.card_down_call_mean.set_data(f"₹ {down.call_mean:.2f}", f"Median: ₹ {down.call_median:.2f}")
            self.card_down_call_best.set_data(f"₹ {down.call_max:.2f}", f"Worst Case: ₹ {down.call_min:.2f}")
            self.card_down_call_range.set_data(f"₹ {down.call_p05:.2f} - ₹ {down.call_p95:.2f}", f"Std Dev: ₹ {down.call_std:.2f}")

            self.card_down_put_mean.set_data(f"₹ {down.put_mean:.2f}", f"Median: ₹ {down.put_median:.2f}")
            self.card_down_put_best.set_data(f"₹ {down.put_max:.2f}", f"Worst Case: ₹ {down.put_min:.2f}")
            self.card_down_put_range.set_data(f"₹ {down.put_p05:.2f} - ₹ {down.put_p95:.2f}", f"Std Dev: ₹ {down.put_std:.2f}")
        else:
            for card in [self.card_down_call_mean, self.card_down_call_best, self.card_down_call_range,
                         self.card_down_put_mean, self.card_down_put_best, self.card_down_put_range]:
                card.set_data("No Hits", "Target not reached")

    def plot_paths(self, res: dict):
        self.fig_paths.clear()

        cfg: SimulationConfig = res['config']
        paths      = res['paths_sample']
        time_mins  = res['time_axis_minutes']
        ps         = res['sf_path_stats']     # SF projected high/low stats

        # ── Layout: main chart (left) + stats panel (right) ──────────────────
        gs = self.fig_paths.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.04)
        ax   = self.fig_paths.add_subplot(gs[0])
        ax_s = self.fig_paths.add_subplot(gs[1])

        # ── CHART ─────────────────────────────────────────────────────────────
        ax.set_facecolor('#0D1117')
        for i in range(min(50, len(paths))):
            ax.plot(time_mins, paths[i], color='#4A5568', alpha=0.30, linewidth=0.7)

        # Barrier / reference lines
        ax.axhline(cfg.opening_sf,     color='#ECC94B', linestyle='--', linewidth=1.6,
                   label=f"Opening SF: {cfg.opening_sf:.1f}")
        ax.axhline(cfg.upside_target,  color='#48BB78', linestyle='-',  linewidth=2.0,
                   label=f"Upside Target: {cfg.upside_target:.1f}")
        ax.axhline(cfg.downside_target,color='#F56565', linestyle='-',  linewidth=2.0,
                   label=f"Downside Target: {cfg.downside_target:.1f}")

        # Projected High / Low mean lines (from sim stats)
        ax.axhline(ps['high_mean'],  color='#68D391', linestyle=':', linewidth=1.4,
                   label=f"Avg High: {ps['high_mean']:.1f}")
        ax.axhline(ps['low_mean'],   color='#FC8181', linestyle=':', linewidth=1.4,
                   label=f"Avg Low:  {ps['low_mean']:.1f}")
        ax.axhline(ps['high_median'],color='#9AE6B4', linestyle='-.', linewidth=1.0,
                   label=f"Med High: {ps['high_median']:.1f}")
        ax.axhline(ps['low_median'], color='#FEB2B2', linestyle='-.', linewidth=1.0,
                   label=f"Med Low:  {ps['low_median']:.1f}")

        # Shaded band between mean high and mean low
        ax.axhspan(ps['low_mean'], ps['high_mean'], alpha=0.06, color='#63B3ED')

        ax.set_title("Synthetic Future — Monte Carlo Trajectories & Barrier Levels",
                     color='#E2E8F0', fontsize=12, fontweight='bold', pad=8)
        ax.set_xlabel("Elapsed Market Time (Minutes)  [385 min = 1 session]",
                      color='#A0AEC0', fontsize=11)
        ax.set_ylabel("Synthetic Future (SF)", color='#A0AEC0', fontsize=11)
        ax.tick_params(colors='#CBD5E0', labelsize=10)
        for sp in ax.spines.values(): sp.set_color('#2D3748')
        ax.grid(True, linestyle=':', alpha=0.25, color='#4A5568')
        ax.legend(loc='upper left', facecolor='#1A202C', edgecolor='#2D3748',
                  labelcolor='#E2E8F0', fontsize=9, framealpha=0.9)

        # ── STATS PANEL ───────────────────────────────────────────────────────
        ax_s.set_facecolor('#0D1117')
        ax_s.axis('off')

        col_h = '#68D391'   # green for high
        col_l = '#FC8181'   # red for low
        col_t = '#E2E8F0'   # white for headers
        col_m = '#A0AEC0'   # muted for labels

        rows = [
            # (label, value_str, color)
            ("── PROJECTED HIGH ──",        "",                             col_h),
            ("Average High",    f"{ps['high_mean']:.2f}",   col_h),
            ("Median High",     f"{ps['high_median']:.2f}", col_h),
            ("Best High (Max)", f"{ps['high_best']:.2f}",   '#48BB78'),
            ("Worst High (Min)",f"{ps['high_worst']:.2f}",  col_m),
            ("5th %ile High",   f"{ps['high_p05']:.2f}",    col_m),
            ("95th %ile High",  f"{ps['high_p95']:.2f}",    col_m),
            ("",                "",                          col_t),
            ("── PROJECTED LOW ──",         "",                             col_l),
            ("Average Low",     f"{ps['low_mean']:.2f}",    col_l),
            ("Median Low",      f"{ps['low_median']:.2f}",  col_l),
            ("Best Low (Min)",  f"{ps['low_best']:.2f}",    '#F56565'),
            ("Worst Low (Max)", f"{ps['low_worst']:.2f}",   col_m),
            ("5th %ile Low",    f"{ps['low_p05']:.2f}",     col_m),
            ("95th %ile Low",   f"{ps['low_p95']:.2f}",     col_m),
        ]

        n = len(rows)
        for idx, (label, val, color) in enumerate(rows):
            y = 1.0 - (idx + 0.5) / n
            if val == "":
                # Section header
                ax_s.text(0.5, y, label, transform=ax_s.transAxes,
                          ha='center', va='center', color=color,
                          fontsize=10, fontweight='bold',
                          fontfamily='monospace')
            else:
                ax_s.text(0.02, y, label, transform=ax_s.transAxes,
                          ha='left', va='center', color=col_m,
                          fontsize=10, fontfamily='monospace')
                ax_s.text(0.98, y, val, transform=ax_s.transAxes,
                          ha='right', va='center', color=color,
                          fontsize=11, fontweight='bold', fontfamily='monospace')

        # Border around stats panel
        for sp in ax_s.spines.values():
            sp.set_visible(True)
            sp.set_color('#2D3748')
            sp.set_linewidth(1.2)

        ax_s.set_title("Path Statistics", color='#63B3ED',
                       fontsize=11, fontweight='bold', pad=6)

        self.fig_paths.tight_layout()
        self.canvas_paths.draw()

    def plot_distributions(self, res: dict):
        self.fig_dist.clear()

        up: TargetHitStats = res['upside_stats']
        down: TargetHitStats = res['downside_stats']

        # Use constrained_layout for automatic clean spacing - no overlapping labels
        self.fig_dist.set_constrained_layout(True)
        axes = self.fig_dist.subplots(2, 2)

        plots = [
            (axes[0, 0], up.hit_call_prices,   up.call_mean,   up.call_max,
             f"Calls @ Upside {up.target_level:.0f}",   '#38A169'),
            (axes[0, 1], up.hit_put_prices,    up.put_mean,    up.put_max,
             f"Puts @ Upside {up.target_level:.0f}",    '#E53E3E'),
            (axes[1, 0], down.hit_call_prices, down.call_mean, down.call_max,
             f"Calls @ Downside {down.target_level:.0f}", '#38A169'),
            (axes[1, 1], down.hit_put_prices,  down.put_mean,  down.put_max,
             f"Puts @ Downside {down.target_level:.0f}",  '#E53E3E'),
        ]

        for ax, data, mean_val, best_val, title, col in plots:
            ax.set_facecolor('#171923')
            ax.set_title(title, color='#E2E8F0', fontsize=12, fontweight='bold', pad=6)
            # Short, non-overlapping axis labels
            ax.set_xlabel("Option Price (₹)", color='#A0AEC0', fontsize=11, fontweight='bold', labelpad=5)
            ax.set_ylabel("Paths (Count)", color='#A0AEC0', fontsize=11, fontweight='bold', labelpad=5)
            # Rotate & format x tick labels so they never collide
            ax.tick_params(axis='x', colors='#CBD5E0', labelsize=10, rotation=30)
            ax.tick_params(axis='y', colors='#CBD5E0', labelsize=10)
            # Format x-axis numbers as plain integers (no scientific notation)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}'))
            for spine in ax.spines.values():
                spine.set_color('#2D3748')
            ax.grid(True, linestyle=':', alpha=0.3, color='#4A5568')

            if len(data) > 0:
                ax.hist(data, bins=30, color=col, alpha=0.75, edgecolor='#0D1117')
                ax.axvline(mean_val, color='#ECC94B', linestyle='--', linewidth=2,
                           label=f"Mean ₹{mean_val:.1f}")
                ax.axvline(best_val, color='#63B3ED', linestyle=':', linewidth=2,
                           label=f"Best ₹{best_val:.1f}")
                ax.legend(loc='upper right', facecolor='#1A202C', edgecolor='#4A5568',
                          labelcolor='#E2E8F0', fontsize=10, framealpha=0.9)
            else:
                ax.text(0.5, 0.5, "No hits", transform=ax.transAxes,
                        ha='center', va='center', color='#718096', fontsize=12)

        self.canvas_dist.draw()

    def populate_table(self):
        if not self.sim_results:
            return

        up: TargetHitStats = self.sim_results['upside_stats']
        down: TargetHitStats = self.sim_results['downside_stats']
        cfg: SimulationConfig = self.sim_results['config']

        filter_mode = self.combo_table_filter.currentIndex()
        rows = []

        if filter_mode in (0, 1) and up.hit_count > 0:
            for i in range(up.hit_count):
                path_id = up.hit_indices[i]
                el_mins = up.hit_elapsed_minutes[i]
                el_dte = (el_mins / cfg.minutes_per_day)
                rem_dte = max(0.0, cfg.dte_days - el_dte)
                c_p = up.hit_call_prices[i]
                p_p = up.hit_put_prices[i]
                rows.append((path_id, "Upside Hit", up.target_level, el_mins, el_dte, rem_dte, c_p, p_p))

        if filter_mode in (0, 2) and down.hit_count > 0:
            for i in range(down.hit_count):
                path_id = down.hit_indices[i]
                el_mins = down.hit_elapsed_minutes[i]
                el_dte = (el_mins / cfg.minutes_per_day)
                rem_dte = max(0.0, cfg.dte_days - el_dte)
                c_p = down.hit_call_prices[i]
                p_p = down.hit_put_prices[i]
                rows.append((path_id, "Downside Hit", down.target_level, el_mins, el_dte, rem_dte, c_p, p_p))

        display_limit = 1500
        disp_rows = rows[:display_limit]
        self.table.setRowCount(len(disp_rows))

        for r_idx, r in enumerate(disp_rows):
            path_id, target_type, target_lvl, el_mins, el_dte, rem_dte, c_p, p_p = r

            it_id = QTableWidgetItem(f"#{path_id}")
            it_type = QTableWidgetItem(target_type)
            if "Upside" in target_type:
                it_type.setForeground(QColor("#68D391"))
            else:
                it_type.setForeground(QColor("#FC8181"))

            it_lvl = QTableWidgetItem(f"{target_lvl:.1f}")
            it_mins = QTableWidgetItem(f"{el_mins:.1f}")
            it_eldte = QTableWidgetItem(f"{el_dte:.3f}")
            it_remdte = QTableWidgetItem(f"{rem_dte:.3f}")
            it_call = QTableWidgetItem(f"₹ {c_p:.2f}")
            it_put = QTableWidgetItem(f"₹ {p_p:.2f}")

            for it in [it_id, it_type, it_lvl, it_mins, it_eldte, it_remdte, it_call, it_put]:
                it.setTextAlignment(Qt.AlignCenter)

            self.table.setItem(r_idx, 0, it_id)
            self.table.setItem(r_idx, 1, it_type)
            self.table.setItem(r_idx, 2, it_lvl)
            self.table.setItem(r_idx, 3, it_mins)
            self.table.setItem(r_idx, 4, it_eldte)
            self.table.setItem(r_idx, 5, it_remdte)
            self.table.setItem(r_idx, 6, it_call)
            self.table.setItem(r_idx, 7, it_put)

        total_matches = len(rows)
        self.lbl_table_count.setText(f"Showing {len(disp_rows)} of {total_matches} simulated hits (Use Export to save all rows)")

    def export_csv(self):
        if not self.sim_results:
            QMessageBox.information(self, "No Data", "Please run a simulation first before exporting.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Export Simulations to CSV", "monte_carlo_simulations.csv", "CSV Files (*.csv)")
        if not path:
            return

        up: TargetHitStats = self.sim_results['upside_stats']
        down: TargetHitStats = self.sim_results['downside_stats']
        cfg: SimulationConfig = self.sim_results['config']

        try:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Path_ID", "Target_Event", "Target_Price", "Elapsed_Market_Minutes",
                    "Elapsed_DTE_Days", "Remaining_DTE_Days", "Remaining_T_Years",
                    "Call_Strike", "Call_Price", "Put_Strike", "Put_Price", "IV_at_Hit_Pct"
                ])

                for i in range(up.hit_count):
                    el_mins = up.hit_elapsed_minutes[i]
                    el_dte = (el_mins / cfg.minutes_per_day)
                    rem_dte = max(0.0, cfg.dte_days - el_dte)
                    writer.writerow([
                        up.hit_indices[i], "Upside Hit", up.target_level, f"{el_mins:.2f}",
                        f"{el_dte:.4f}", f"{rem_dte:.4f}", f"{(rem_dte / 365.0):.6f}",
                        cfg.call_strike, f"{up.hit_call_prices[i]:.2f}",
                        cfg.put_strike, f"{up.hit_put_prices[i]:.2f}", f"{up.iv_at_hit_pct:.2f}"
                    ])

                for i in range(down.hit_count):
                    el_mins = down.hit_elapsed_minutes[i]
                    el_dte = (el_mins / cfg.minutes_per_day)
                    rem_dte = max(0.0, cfg.dte_days - el_dte)
                    writer.writerow([
                        down.hit_indices[i], "Downside Hit", down.target_level, f"{el_mins:.2f}",
                        f"{el_dte:.4f}", f"{rem_dte:.4f}", f"{(rem_dte / 365.0):.6f}",
                        cfg.call_strike, f"{down.hit_call_prices[i]:.2f}",
                        cfg.put_strike, f"{down.hit_put_prices[i]:.2f}", f"{down.iv_at_hit_pct:.2f}"
                    ])

            QMessageBox.information(self, "Export Successful", f"Saved all simulations to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"Could not write CSV:\n{str(e)}")


def main():
    app = QApplication(sys.argv)
    window = OptionsSimApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

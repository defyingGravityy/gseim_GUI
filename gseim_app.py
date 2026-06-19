import sys
import os
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

try:
    from PyQt6 import QtCore, QtWidgets
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap, QDoubleValidator
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog,
        QPushButton, QLabel, QListWidget, QListWidgetItem,
        QTableWidget, QTableWidgetItem, QAbstractItemView,
        QFileDialog, QMessageBox, QCheckBox, QComboBox,
        QLineEdit, QColorDialog, QHeaderView, QSizePolicy,
        QVBoxLayout, QHBoxLayout, QFormLayout, QFrame,
        QScrollArea,
    )

    Checked   = QtCore.Qt.CheckState.Checked
    Unchecked = QtCore.Qt.CheckState.Unchecked
    UserCheckable   = QtCore.Qt.ItemFlag.ItemIsUserCheckable
    ItemEnabled     = QtCore.Qt.ItemFlag.ItemIsEnabled
    NoSelection     = QAbstractItemView.SelectionMode.NoSelection
    SingleSelection = QAbstractItemView.SelectionMode.SingleSelection
    StretchLast     = QHeaderView.ResizeMode.Stretch
    ResizeToContents = QHeaderView.ResizeMode.ResizeToContents
    ScrollBarAlwaysOff = QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff
    NoFrame = QFrame.Shape.NoFrame
except ImportError:
    from PyQt5 import QtCore, QtWidgets
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor, QFont, QIcon, QPixmap, QDoubleValidator
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QDialog,
        QPushButton, QLabel, QListWidget, QListWidgetItem,
        QTableWidget, QTableWidgetItem, QAbstractItemView,
        QFileDialog, QMessageBox, QCheckBox, QComboBox,
        QLineEdit, QColorDialog, QHeaderView, QSizePolicy,
        QVBoxLayout, QHBoxLayout, QFormLayout, QFrame,
        QScrollArea,
    )
    Checked   = QtCore.Qt.Checked
    Unchecked = QtCore.Qt.Unchecked
    UserCheckable    = QtCore.Qt.ItemIsUserCheckable
    ItemEnabled      = QtCore.Qt.ItemIsEnabled
    NoSelection      = QAbstractItemView.NoSelection
    SingleSelection  = QAbstractItemView.SingleSelection
    StretchLast      = QHeaderView.Stretch
    ResizeToContents = QHeaderView.ResizeToContents
    ScrollBarAlwaysOff = QtCore.Qt.ScrollBarAlwaysOff
    NoFrame = QFrame.NoFrame

# Matplotlib inside Qt 
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
try:
    from matplotlib.backends.backend_qtagg import (
        FigureCanvas, NavigationToolbar2QT as NavToolbar)
except ImportError:
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvas, NavigationToolbar2QT as NavToolbar)


@dataclass
class OutputBlock:
    dat_filename: str           
    variables: list[str] = field(default_factory=list)  
    dat_path: Path | None = None  


class GseimInFileError(ValueError):
    """Raised when a .in file can't be parsed the way we expect."""


def parse_in_file(in_file_path) -> tuple[int, list[OutputBlock]]:
    in_file_path = Path(in_file_path).expanduser()
    if not in_file_path.exists():
        raise GseimInFileError(f".in file not found: {in_file_path}")

    lines = in_file_path.read_text().splitlines()
    base_dir = in_file_path.parent

    n_solve_blocks = sum(1 for ln in lines if "begin_solve" in ln)

    output_blocks: list[OutputBlock] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if "begin_output" in line:
            block_start = i
            next_line = lines[i + 1] if i + 1 < n else ""
            if ".dat" in next_line:
                eq_pos = next_line.index("=")
                dat_end = next_line.index(".dat") + len(".dat")
                dat_filename = next_line[eq_pos + 1:dat_end].strip()

                j = i + 1
                variables: list[str] = []
                while j < n and "end_output" not in lines[j]:
                    if "variables:" in lines[j]:
                        after_colon = lines[j].split(":", 1)[1]
                        variables = after_colon.strip().split()
                    j += 1
                if j >= n:
                    raise GseimInFileError(
                        f"begin_output at line {block_start + 1} has no matching end_output"
                    )

                output_blocks.append(OutputBlock(
                    dat_filename=dat_filename,
                    variables=variables,
                    dat_path=(base_dir / dat_filename).resolve(),
                ))
                i = j
        i += 1

    return n_solve_blocks, output_blocks


def load_dat(dat_path) -> np.ndarray:
    dat_path = Path(dat_path).expanduser()
    if not dat_path.exists():
        raise FileNotFoundError(f".dat file not found: {dat_path}")

    data = np.loadtxt(dat_path)
    if data.size == 0:
        raise ValueError(f".dat file has no data: {dat_path}")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def column_labels(variables: list[str], n_col: int) -> list[str]:
    labels = list(variables)
    if len(labels) == n_col - 1:
        labels.insert(0, "time")
    if len(labels) != n_col:
        labels = [f"col{i}" for i in range(n_col)]
    return labels

class TrimmedToolbar(NavToolbar):
    """Keep only the four toolbar buttons we actually need."""
    # toolitems = [t for t in NavToolbar.toolitems
                #  if t[0] in ("Home", "Pan", "Zoom", "Save")]
    toolitems =toolitems = NavToolbar.toolitems 

def fourier_coeff(t, x, t_start, t_end, n_fourier):

     coeff = [0.0]*n_fourier
     sum_fourier_real = [0.0]*n_fourier
     sum_fourier_imag = [0.0]*n_fourier
     sum_fourier_mag = [0.0]*n_fourier
     omg = [0.0]*n_fourier
     sum_rms_1 = 0.0

     T = t_end - t_start
     a2 = 0.5*T
     f_hz = 1.0/T
     for i in range(n_fourier):
         omg[i] = 2.0*float(i)*np.pi/T

     t_last = t[0];
     y_last = x[0];

     n = len(t)

     for i in range(n):
         t0 = t[i]
         y0 = x[i]

         if t_last <= t_end:
             if t_last >= t_start:
                 if t0 <= t_end:
                     delta_t = t0 - t_last
                     for j in range(n_fourier):
                         wt1 = omg[j]*t_last
                         wt2 = omg[j]*t0
                         z1 = y_last
                         z2 = y0
                         sum_fourier_real[j] += 0.5*delta_t*(np.cos(wt1)*z1 + np.cos(wt2)*z2)
                         sum_fourier_imag[j] += 0.5*delta_t*(np.sin(wt1)*z1 + np.sin(wt2)*z2)
                     sum_rms_1 += 0.5*delta_t*(y_last*y_last + y0*y0)
                 else:
                     y1 = y_last + ((y0-y_last)/(t0-t_last))*(t_end-t_last)
                     delta_t = t_end-t_last
                     for j in range(n_fourier):
                         wt1 = omg[j]*t_last
                         wt2 = omg[j]*t_end
                         z1 = y_last
                         z2 = y1
                         sum_fourier_real[j] += 0.5*delta_t*(np.cos(wt1)*z1 + np.cos(wt2)*z2)
                         sum_fourier_imag[j] += 0.5*delta_t*(np.sin(wt1)*z1 + np.sin(wt2)*z2)
                     sum_rms_1 += 0.5*delta_t*(y_last*y_last + y1*y1)
             else:
                 if t0 > t_start:
                     y1 = y_last + ((y0-y_last)/(t0-t_last))*(t_start-t_last)
                     delta_t = t0-t_start
                     for j in range(n_fourier):
                         wt1 = omg[j]*t_start
                         wt2 = omg[j]*t0
                         z1 = y1
                         z2 = y0
                         sum_fourier_real[j] += 0.5*delta_t*(np.cos(wt1)*z1 + np.cos(wt2)*z2)
                         sum_fourier_imag[j] += 0.5*delta_t*(np.sin(wt1)*z1 + np.sin(wt2)*z2)
                     sum_rms_1 += 0.5*delta_t*(y1*y1 + y0*y0)
         t_last = t0
         y_last = y0

     for i in range(n_fourier):
         b1 = sum_fourier_real[i]
         b2 = sum_fourier_imag[i]
         sum_fourier_mag[i] = np.sqrt((b1*b1)+(b2*b2))/a2

     coeff[0] = 0.5*sum_fourier_mag[0]
     for i in range(1, n_fourier):
         coeff[i] = sum_fourier_mag[i]

     arg1 = 2.0*(sum_rms_1/T) - 2.0*coeff[0]**2 - coeff[1]**2
     thd = np.sqrt(arg1)/coeff[1]

     return coeff, thd


@dataclass
class LineStyling:
    label:      str   = ""
    line_style: str   = "solid"  
    draw_style: str   = "default" # default / steps-post / steps-pre / steps-mid
    width:      float = 1.0
    color:      str   = "royalblue"
    marker:     str   = ""       
    marker_size:       float = 3.0
    marker_edge_color: str   = "black"
    marker_face_color: str   = "white"
    multi_scale:  str = "linear"  


@dataclass
class TitleSettings:
    text:    str  = ""
    loc:     str  = "center"   # left / center / right
    enabled: bool = True


@dataclass
class GridSettings:
    enabled:    bool  = True
    color:      str   = "lightgrey"
    line_style: str   = "solid"
    width:      float = 0.7
    which:      str   = "both"   # major / minor / both
    axis:       str   = "both"   # x / y / both


@dataclass
class MultiPlotSettings:
    enabled: bool = False   # if True, stack one subplot per Y variable


@dataclass
class FourierSettings:
    enabled:   bool  = False
    n_fourier: int   = 10    # number of harmonics to compute
    t_start:   float = 0.0
    t_end:     float = 0.0


@dataclass
class AvgRmsSettings:
    avg:    bool  = False
    rms:    bool  = False
    period: float = 0.0


@dataclass
class PowerSettings:
    enabled:  bool  = False
    v_name:   str   = ""    # name of the voltage column
    i_name:   str   = ""    # name of the current column
    period:   float = 0.0


@dataclass
class AxesSettings:
    # X axis
    x_scale: str   = "linear"
    x_label: str   = ""
    x_min:   str   = ""     
    x_max:   str   = ""
    x_sn_lo: int   = -3    
    x_sn_hi: int   =  3     
    x_sn:    bool  = True  
    # Left Y axis
    y_scale: str   = "linear"
    y_label: str   = ""
    y_min:   str   = ""
    y_max:   str   = ""
    y_sn_lo: int   = -3
    y_sn_hi: int   =  3
    y_sn:    bool  = True
    y_scale2: str  = "linear"
    y_label2: str  = ""
    y_min2:   str  = ""
    y_max2:   str  = ""
    y_sn_lo2: int  = -3
    y_sn_hi2: int  =  3
    y_sn2:    bool = True


@dataclass
class LegendSettings:
    location:       str   = "best"
    frame:          bool  = True
    fontsize:       int   = 10
    title:          str   = ""
    marker_first:   bool  = True
    marker_scale:   float = 1.0
    label_spacing:  float = 0.5
    column_spacing: float = 2.0


@dataclass
class TickSettings:
    enabled:   bool  = True
    direction: str   = "out"   
    rotation:  int   = 0
    positions: list  = field(default_factory=list)
    labels:    list  = field(default_factory=list)


#HELPER: base class for all popup dialogs

class BasePopup(QMainWindow):
    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.content_widget = QWidget()
        self.content_layout = QFormLayout(self.content_widget)
        self.content_layout.setContentsMargins(12, 12, 12, 8)
        self.content_layout.setSpacing(8)

        btn_row = QWidget()
        btn_layout = QHBoxLayout(btn_row)
        btn_layout.setContentsMargins(8, 4, 8, 8)

        self.apply_btn  = QPushButton("Apply")
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn     = QPushButton("Ok")
        for b in (self.apply_btn, self.cancel_btn, self.ok_btn):
            btn_layout.addWidget(b)

        self.apply_btn.clicked.connect(self.apply)
        self.cancel_btn.clicked.connect(self.close)
        self.ok_btn.clicked.connect(self._ok)  

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.addWidget(self.content_widget)
        root_layout.addWidget(btn_row)
        self.setCentralWidget(root)

    def _ok(self):
        self.apply()    
        self.close()

    def showEvent(self, event):
        self.load_settings()
        super().showEvent(event)

    def load_settings(self):
        pass

    def apply(self):
        pass


#POPUP DIALOGS

def _color_button(initial_color: str) -> tuple:
    state = {"color": QColor(initial_color)}
    btn = QPushButton()
    btn.setFixedSize(28, 22)

    def _refresh():
        px = QPixmap(16, 16)
        px.fill(state["color"])
        btn.setIcon(QIcon(px))

    def _pick():
        c = QColorDialog.getColor(state["color"])
        if c.isValid():
            state["color"] = c
            _refresh()

    btn.clicked.connect(_pick)
    _refresh()
    return btn, state


class LinePropPopup(BasePopup):
    def __init__(self, main_win):
        super().__init__("Line Properties")
        self.main_win = main_win
        self.resize(420, 280)

        # variable picker 
        self.var_combo = QComboBox()
        self.var_combo.currentIndexChanged.connect(self._var_changed)
        self.content_layout.addRow("Variable:", self.var_combo)

        self.legend_label = QLineEdit()
        self.content_layout.addRow("Legend label:", self.legend_label)

        #  line 
        self.line_style_combo = QComboBox()
        for s in ("solid", "dashed", "dotted", "dashdot", "None"):
            self.line_style_combo.addItem(s)
        self.content_layout.addRow("Line style:", self.line_style_combo)

        self.draw_style_combo = QComboBox()
        for s in ("default", "steps-post", "steps-pre", "steps-mid"):
            self.draw_style_combo.addItem(s)
        self.content_layout.addRow("Draw style:", self.draw_style_combo)

        self.line_width = QLineEdit("1.0")
        self.line_width.setValidator(QDoubleValidator(0.1, 20.0, 2))
        self.content_layout.addRow("Line width:", self.line_width)

        self.line_color_btn, self._line_color = _color_button("royalblue")
        self.content_layout.addRow("Line color:", self.line_color_btn)

        #  marker
        self.marker_combo = QComboBox()
        for m in ("", ".", "o", "x", "+", "D", "d", "s", "*", "v", "^", "<", ">"):
            self.marker_combo.addItem(m)
        self.content_layout.addRow("Marker style:", self.marker_combo)

        self.marker_size = QLineEdit("3.0")
        self.marker_size.setValidator(QDoubleValidator(0.1, 50.0, 2))
        self.content_layout.addRow("Marker size:", self.marker_size)

        self.edge_color_btn, self._edge_color = _color_button("black")
        self.content_layout.addRow("Marker edge color:", self.edge_color_btn)

        self.face_color_btn, self._face_color = _color_button("white")
        self.content_layout.addRow("Marker face color:", self.face_color_btn)

        # multi-plot scale
        self.multi_scale_combo = QComboBox()
        self.multi_scale_combo.addItem("linear")
        self.multi_scale_combo.addItem("log")
        self.content_layout.addRow("Scale (multiplot):", self.multi_scale_combo)

    def load_settings(self):
        # Rebuild the variable dropdown from whatever is currently checked as Y
        self.var_combo.blockSignals(True)
        self.var_combo.clear()
        mw = self.main_win
        all_y_rows = mw.y_left_rows + mw.y_right_rows
        for r in all_y_rows:
            self.var_combo.addItem(mw.labels[r])
        self.var_combo.blockSignals(False)
        if self.var_combo.count() > 0:
            self._load_row(all_y_rows[0] if all_y_rows else 0)

    def _var_changed(self, idx):
        mw = self.main_win
        all_y_rows = mw.y_left_rows + mw.y_right_rows
        if 0 <= idx < len(all_y_rows):
            self._load_row(all_y_rows[idx])

    def _load_row(self, row):
        s = self.main_win.line_styles[row]
        self.legend_label.setText(s.label)
        self.line_style_combo.setCurrentText(s.line_style)
        self.draw_style_combo.setCurrentText(s.draw_style)
        self.line_width.setText(str(s.width))
        self._line_color["color"] = QColor(s.color)
        px = QPixmap(16, 16); px.fill(self._line_color["color"])
        self.line_color_btn.setIcon(QIcon(px))
        self.marker_combo.setCurrentText(s.marker)
        self.marker_size.setText(str(s.marker_size))
        self._edge_color["color"] = QColor(s.marker_edge_color)
        px.fill(self._edge_color["color"]); self.edge_color_btn.setIcon(QIcon(px))
        self._face_color["color"] = QColor(s.marker_face_color)
        px.fill(self._face_color["color"]); self.face_color_btn.setIcon(QIcon(px))
        self.multi_scale_combo.setCurrentText(s.multi_scale)

    def apply(self):
        mw = self.main_win
        all_y_rows = mw.y_left_rows + mw.y_right_rows
        idx = self.var_combo.currentIndex()
        if not all_y_rows or idx < 0:
            return
        row = all_y_rows[idx]
        s = mw.line_styles[row]
        s.label       = self.legend_label.text()
        s.line_style  = self.line_style_combo.currentText()
        s.draw_style  = self.draw_style_combo.currentText()
        s.width       = float(self.line_width.text() or "1.0")
        s.color       = self._line_color["color"].name()
        s.marker      = self.marker_combo.currentText()
        s.marker_size = float(self.marker_size.text() or "3.0")
        s.marker_edge_color = self._edge_color["color"].name()
        s.marker_face_color = self._face_color["color"].name()
        s.multi_scale = self.multi_scale_combo.currentText()


class MultiPlotPopup(BasePopup):
    def __init__(self, main_win):
        super().__init__("Multi-Plot")
        self.main_win = main_win
        self.resize(260, 120)
        self.check = QCheckBox()
        self.content_layout.addRow("Stack subplots:", self.check)

    def load_settings(self):
        self.check.setChecked(self.main_win.multiplot.enabled)

    def apply(self):
        mw = self.main_win
        if self.check.isChecked() and mw.x_col != 0:
            QMessageBox.warning(self, "Multi-Plot", "X axis must be 'time' for multi-plot mode.")
            self.check.setChecked(False)
            return
        mw.multiplot.enabled = self.check.isChecked()


class FourierPopup(BasePopup):
    def __init__(self, main_win):
        super().__init__("Fourier Analysis")
        self.main_win = main_win
        self.resize(300, 200)
        self.check = QCheckBox()
        self.n_edit  = QLineEdit("10")
        self.t0_edit = QLineEdit("0")
        self.t1_edit = QLineEdit("0")
        self.content_layout.addRow("Enabled:", self.check)
        self.content_layout.addRow("Number of harmonics:", self.n_edit)
        self.content_layout.addRow("t start:", self.t0_edit)
        self.content_layout.addRow("t end:",   self.t1_edit)

    def load_settings(self):
        f = self.main_win.fourier
        self.check.setChecked(f.enabled)
        self.n_edit.setText(str(f.n_fourier))
        self.t0_edit.setText(str(f.t_start))
        self.t1_edit.setText(str(f.t_end))

    def apply(self):
        mw = self.main_win
        f  = mw.fourier
        f.enabled   = self.check.isChecked()
        if not f.enabled:
            return
        t0 = float(self.t0_edit.text() or "0")
        t1 = float(self.t1_edit.text() or "0")
        if t0 >= t1:
            QMessageBox.warning(self, "Fourier", "t start must be less than t end.")
            return
        if mw.x_col != 0:
            QMessageBox.warning(self, "Fourier", "X axis must be 'time' for Fourier analysis.")
            return
        f.n_fourier = int(self.n_edit.text() or "10")
        f.t_start   = t0
        f.t_end     = t1
        mw._run_fourier()


class AvgRmsPopup(BasePopup):
    """Compute a rolling average and/or rolling RMS over one period at a time."""
    def __init__(self, main_win):
        super().__init__("Average / RMS")
        self.main_win = main_win
        self.resize(280, 160)
        self.avg_check = QCheckBox()
        self.rms_check = QCheckBox()
        self.period_edit = QLineEdit()
        self.content_layout.addRow("Compute average:", self.avg_check)
        self.content_layout.addRow("Compute RMS:",     self.rms_check)
        self.content_layout.addRow("Period:",          self.period_edit)

    def load_settings(self):
        a = self.main_win.avgrms
        self.avg_check.setChecked(a.avg)
        self.rms_check.setChecked(a.rms)
        self.period_edit.setText(str(a.period) if a.period else "")

    def apply(self):
        mw = self.main_win
        a  = mw.avgrms
        a.avg = self.avg_check.isChecked()
        a.rms = self.rms_check.isChecked()
        if not (a.avg or a.rms):
            return
        try:
            T = float(self.period_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Avg/RMS", "Enter a valid period."); return
        if T <= 0:
            QMessageBox.warning(self, "Avg/RMS", "Period must be positive."); return
        if mw.x_col != 0:
            QMessageBox.warning(self, "Avg/RMS", "X axis must be 'time' for avg/rms."); return
        a.period = T
        mw._run_avgrms()


class PowerPopup(BasePopup):
    def __init__(self, main_win):
        super().__init__("Power Computation")
        self.main_win = main_win
        self.resize(300, 180)
        self.check      = QCheckBox()
        self.v_combo    = QComboBox()
        self.i_combo    = QComboBox()
        self.period_edit = QLineEdit()
        self.content_layout.addRow("Enabled:",  self.check)
        self.content_layout.addRow("Voltage column:", self.v_combo)
        self.content_layout.addRow("Current column:", self.i_combo)
        self.content_layout.addRow("Period:",    self.period_edit)

    def load_settings(self):
        p  = self.main_win.power
        mw = self.main_win
        self.check.setChecked(p.enabled)
        self.v_combo.clear(); self.i_combo.clear()
        for lbl in (mw.labels or []):
            self.v_combo.addItem(lbl)
            self.i_combo.addItem(lbl)
        if p.v_name in mw.labels:
            self.v_combo.setCurrentText(p.v_name)
        if p.i_name in mw.labels:
            self.i_combo.setCurrentText(p.i_name)
        self.period_edit.setText(str(p.period) if p.period else "")

    def apply(self):
        mw = self.main_win
        p  = mw.power
        p.enabled = self.check.isChecked()
        if not p.enabled:
            return
        try:
            T = float(self.period_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Power", "Enter a valid period."); return
        if T <= 0:
            QMessageBox.warning(self, "Power", "Period must be positive."); return
        p.v_name = self.v_combo.currentText()
        p.i_name = self.i_combo.currentText()
        p.period = T
        mw._run_power()


class GridPopup(BasePopup):
    def __init__(self, main_win):
        super().__init__("Grid")
        self.main_win = main_win
        self.resize(280, 220)
        self.check      = QCheckBox()
        self.color_btn, self._color = _color_button("lightgrey")
        self.style_combo = QComboBox()
        for s in ("solid", "dashed", "dotted", "dashdot"):
            self.style_combo.addItem(s)
        self.width_edit = QLineEdit("0.7")
        self.width_edit.setValidator(QDoubleValidator(0.1, 5.0, 2))
        self.which_combo = QComboBox()
        for s in ("both", "major", "minor"): self.which_combo.addItem(s)
        self.axis_combo = QComboBox()
        for s in ("both", "x", "y"): self.axis_combo.addItem(s)

        self.content_layout.addRow("Show grid:",  self.check)
        self.content_layout.addRow("Color:",      self.color_btn)
        self.content_layout.addRow("Line style:", self.style_combo)
        self.content_layout.addRow("Line width:", self.width_edit)
        self.content_layout.addRow("Which ticks:", self.which_combo)
        self.content_layout.addRow("Axis:",       self.axis_combo)

    def load_settings(self):
        g = self.main_win.grid
        self.check.setChecked(g.enabled)
        self._color["color"] = QColor(g.color)
        px = QPixmap(16,16); px.fill(self._color["color"])
        self.color_btn.setIcon(QIcon(px))
        self.style_combo.setCurrentText(g.line_style)
        self.width_edit.setText(str(g.width))
        self.which_combo.setCurrentText(g.which)
        self.axis_combo.setCurrentText(g.axis)

    def apply(self):
        g = self.main_win.grid
        g.enabled    = self.check.isChecked()
        g.color      = self._color["color"].name()
        g.line_style = self.style_combo.currentText()
        g.width      = float(self.width_edit.text() or "0.7")
        g.which      = self.which_combo.currentText()
        g.axis       = self.axis_combo.currentText()


class AxesPopup(BasePopup):
    def __init__(self, main_win):
        super().__init__("Axes Properties")
        self.main_win = main_win
        self.resize(380, 420)

        def row(label, widget):
            self.content_layout.addRow(label, widget)

        self.x_label = QLineEdit()
        self.x_scale = QComboBox(); self.x_scale.addItems(["linear", "log"])
        self.x_min   = QLineEdit(); self.x_max = QLineEdit()

        self.y_label = QLineEdit()
        self.y_scale = QComboBox(); self.y_scale.addItems(["linear", "log"])
        self.y_min   = QLineEdit(); self.y_max = QLineEdit()

        self.y2_label = QLineEdit()
        self.y2_scale = QComboBox(); self.y2_scale.addItems(["linear", "log"])
        self.y2_min   = QLineEdit(); self.y2_max = QLineEdit()

        row("X label:",         self.x_label)
        row("X scale:",         self.x_scale)
        row("X min (blank=auto):", self.x_min)
        row("X max (blank=auto):", self.x_max)
        row("Left Y label:",    self.y_label)
        row("Left Y scale:",    self.y_scale)
        row("Left Y min:",      self.y_min)
        row("Left Y max:",      self.y_max)
        row("Right Y label:",   self.y2_label)
        row("Right Y scale:",   self.y2_scale)
        row("Right Y min:",     self.y2_min)
        row("Right Y max:",     self.y2_max)

    def load_settings(self):
        a = self.main_win.axes
        self.x_label.setText(a.x_label); self.x_scale.setCurrentText(a.x_scale)
        self.x_min.setText(a.x_min);     self.x_max.setText(a.x_max)
        self.y_label.setText(a.y_label); self.y_scale.setCurrentText(a.y_scale)
        self.y_min.setText(a.y_min);     self.y_max.setText(a.y_max)
        self.y2_label.setText(a.y_label2); self.y2_scale.setCurrentText(a.y_scale2)
        self.y2_min.setText(a.y_min2);   self.y2_max.setText(a.y_max2)

    def apply(self):
        a = self.main_win.axes
        a.x_label = self.x_label.text(); a.x_scale = self.x_scale.currentText()
        a.x_min   = self.x_min.text();   a.x_max   = self.x_max.text()
        a.y_label = self.y_label.text(); a.y_scale = self.y_scale.currentText()
        a.y_min   = self.y_min.text();   a.y_max   = self.y_max.text()
        a.y_label2 = self.y2_label.text(); a.y_scale2 = self.y2_scale.currentText()
        a.y_min2  = self.y2_min.text();  a.y_max2  = self.y2_max.text()


class TitlePopup(BasePopup):
    """Set the plot title text, alignment, and whether to show it at all."""
    def __init__(self, main_win):
        super().__init__("Title")
        self.main_win = main_win
        self.resize(300, 150)
        self.check    = QCheckBox()
        self.text     = QLineEdit()
        self.loc      = QComboBox(); self.loc.addItems(["center", "left", "right"])
        self.content_layout.addRow("Show title:", self.check)
        self.content_layout.addRow("Title text:", self.text)
        self.content_layout.addRow("Alignment:",  self.loc)

    def load_settings(self):
        t = self.main_win.title
        self.check.setChecked(t.enabled)
        self.text.setText(t.text)
        self.loc.setCurrentText(t.loc)

    def apply(self):
        t = self.main_win.title
        t.enabled = self.check.isChecked()
        t.text    = self.text.text()
        t.loc     = self.loc.currentText()


class LegendPopup(BasePopup):
    """Control the legend's position, font, and spacing."""
    def __init__(self, main_win):
        super().__init__("Legend")
        self.main_win = main_win
        self.resize(300, 260)
        self.loc      = QComboBox()
        for s in ("best","upper right","upper left","lower right","lower left",
                  "center","upper center","lower center"):
            self.loc.addItem(s)
        self.frame    = QCheckBox()
        self.fontsize = QLineEdit()
        self.title    = QLineEdit()
        self.content_layout.addRow("Location:",  self.loc)
        self.content_layout.addRow("Draw frame:", self.frame)
        self.content_layout.addRow("Font size:",  self.fontsize)
        self.content_layout.addRow("Title:",      self.title)

    def load_settings(self):
        l = self.main_win.legend
        self.loc.setCurrentText(l.location)
        self.frame.setChecked(l.frame)
        self.fontsize.setText(str(l.fontsize))
        self.title.setText(l.title or "")

    def apply(self):
        l = self.main_win.legend
        l.location = self.loc.currentText()
        l.frame    = self.frame.isChecked()
        l.fontsize = int(self.fontsize.text() or "10")
        l.title    = self.title.text() or None


#  PLOT WINDOW 

class PlotWindow(QMainWindow):

    COLORS = ["royalblue", "tomato", "green", "mediumorchid",
              "peru", "crimson", "lightseagreen", "palevioletred"]

    def __init__(self, x, x_label,
                 left_series,  
                 right_series, 
                 settings,     
                 parent=None):
        super().__init__(parent)

        title_parts = [s.label or lbl
                       for _, s in left_series + right_series
                       for lbl in [s.label]]
        self.setWindowTitle(" / ".join(
            [s.label for _, s in left_series+right_series] or ["Plot"]
        ) + f"  vs  {x_label}")

        self.fig    = Figure(figsize=(7, 4), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)

        ax  = self.fig.add_subplot(111)
        ax2 = ax.twinx()
        ax2.set_visible(False)   

        ci = 0
        collected_lines = []
        collected_labels = []

        for y, style in left_series:
            ln, = ax.plot(
                x, y,
                color     = style.color,
                linestyle = style.line_style,
                linewidth = style.width,
                drawstyle = style.draw_style,
                marker    = style.marker or None,
                markersize        = style.marker_size,
                markeredgecolor   = style.marker_edge_color,
                markerfacecolor   = style.marker_face_color,
                label     = style.label,
            )
            collected_lines.append(ln)
            collected_labels.append(style.label)
            ci += 1

        for y, style in right_series:
            ax2.set_visible(True)
            ln, = ax2.plot(
                x, y,
                color     = style.color,
                linestyle = style.line_style,
                linewidth = style.width,
                drawstyle = style.draw_style,
                marker    = style.marker or None,
                markersize        = style.marker_size,
                markeredgecolor   = style.marker_edge_color,
                markerfacecolor   = style.marker_face_color,
                label     = style.label,
            )
            collected_lines.append(ln)
            collected_labels.append(style.label)
            ci += 1

        a = settings["axes"]
        ax.set_xlabel(a.x_label or x_label)
        ax.set_xscale(a.x_scale)
        if a.x_min: ax.set_xlim(left=float(a.x_min))
        if a.x_max: ax.set_xlim(right=float(a.x_max))

        if left_series:
            ax.set_yscale(a.y_scale)
            ax.set_ylabel(a.y_label or (left_series[0][1].label if len(left_series)==1 else ""))
            if a.y_min: ax.set_ylim(bottom=float(a.y_min))
            if a.y_max: ax.set_ylim(top=float(a.y_max))

        if right_series:
            ax2.set_yscale(a.y_scale2)
            ax2.set_ylabel(a.y_label2 or (right_series[0][1].label if len(right_series)==1 else ""))
            if a.y_min2: ax2.set_ylim(bottom=float(a.y_min2))
            if a.y_max2: ax2.set_ylim(top=float(a.y_max2))

        # Apply grid
        g = settings["grid"]
        ax.set_axisbelow(True)
        if g.enabled:
            ax.grid(color=g.color, linestyle=g.line_style,
                    linewidth=g.width, which=g.which, axis=g.axis)

        # Apply title
        t = settings["title"]
        if t.enabled and t.text:
            ax.set_title(t.text, loc=t.loc)

        # Legend
        lg = settings["legend"]
        if collected_labels:
            ax.legend(collected_lines, collected_labels,
                      loc=lg.location, frameon=lg.frame,
                      fontsize=lg.fontsize, title=lg.title or None,
                      markerfirst=lg.marker_first,
                      markerscale=lg.marker_scale,
                      labelspacing=lg.label_spacing,
                      columnspacing=lg.column_spacing)

        central = QWidget()
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        self.addToolBar(TrimmedToolbar(self.canvas, self))
        self.resize(720, 480)


class MultiPlotWindow(QMainWindow):
    def __init__(self, x, x_label, all_series, settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Multi-Plot  vs  " + x_label)

        n = len(all_series)
        self.fig    = Figure(figsize=(7, 2.5*n), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.fig.subplots_adjust(hspace=0.05)

        axes = []
        for i, (y, style, _side) in enumerate(all_series):
            ax = self.fig.add_subplot(n, 1, i+1)
            ax.plot(x, y,
                    color=style.color, linestyle=style.line_style,
                    linewidth=style.width, drawstyle=style.draw_style,
                    marker=style.marker or None,
                    markersize=style.marker_size,
                    label=style.label)
            ax.set_ylabel(style.label)
            if style.multi_scale == "log":
                ax.set_yscale("log")
            ax.set_axisbelow(True)
            g = settings["grid"]
            if g.enabled:
                ax.grid(color=g.color, linestyle=g.line_style,
                        linewidth=g.width, which=g.which, axis=g.axis)
            ax.ticklabel_format(axis="y", style="sci",
                                scilimits=(-2, 2), useMathText=True)
            ax.legend(loc="best", fontsize=9)
            ax.label_outer()  
            axes.append(ax)

        if axes:
            axes[-1].set_xlabel(x_label)

        t = settings["title"]
        if t.enabled and t.text and axes:
            axes[0].set_title(t.text, loc=t.loc)
        central = QWidget()
        layout  = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.setCentralWidget(central)
        self.addToolBar(TrimmedToolbar(self.canvas, self))
        self.resize(720, 300 * n)


# MAIN WINDOW

class MainWindow(QMainWindow):

    COLOR_SET = ["royalblue","tomato","green","mediumorchid",
                 "crimson","lightseagreen","peru","palevioletred","dimgrey"]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GSEIM Waveform Viewer")
        self.resize(340, 680)

        #  runtime state
        self.output_blocks  = []     
        self.data           = None  
        self.labels         = []     
        self.x_col          = None  
        self.y_left_rows    = []     
        self.y_right_rows   = []     
        self.line_styles    = []     
        self.open_windows   = []     

        #  derived-feature file paths
        self.file_fourier  = ""
        self.file_thd      = ""
        self.file_avg      = ""
        self.file_rms      = ""
        self.file_power    = ""

        #  settings objects (one per concern)
        self.title    = TitleSettings()
        self.grid     = GridSettings()
        self.axes     = AxesSettings()
        self.legend   = LegendSettings()
        self.multiplot = MultiPlotSettings()
        self.fourier  = FourierSettings()
        self.avgrms   = AvgRmsSettings()
        self.power    = PowerSettings()
        self.ticks_x  = TickSettings()
        self.ticks_y1 = TickSettings()
        self.ticks_y2 = TickSettings()

        #  popup windows (created once, shown/hidden as needed) 
        self.popup_line    = LinePropPopup(self)
        self.popup_multi   = MultiPlotPopup(self)
        self.popup_fourier = FourierPopup(self)
        self.popup_avgrms  = AvgRmsPopup(self)
        self.popup_power   = PowerPopup(self)
        self.popup_grid    = GridPopup(self)
        self.popup_axes    = AxesPopup(self)
        self.popup_title   = TitlePopup(self)
        self.popup_legend  = LegendPopup(self)

        self._build_ui()

    #  UI construction 

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Load button + file label
        self.load_btn = QPushButton("Load .in file…")
        self.load_btn.clicked.connect(self._load_in_file)
        layout.addWidget(self.load_btn)

        self.file_label = QLabel("No project loaded.")
        self.file_label.setWordWrap(True)
        layout.addWidget(self.file_label)

        # Output files list
        layout.addWidget(QLabel("Output files:"))
        self.output_list = QListWidget()
        self.output_list.setSelectionMode(SingleSelection)
        self.output_list.setMaximumHeight(100)
        self.output_list.currentRowChanged.connect(self._select_output_file)
        layout.addWidget(self.output_list)

        self.shape_label = QLabel("")
        layout.addWidget(self.shape_label)

        # X axis picker
        layout.addWidget(QLabel("X axis:"))
        self.x_list = QListWidget()
        self.x_list.setSelectionMode(SingleSelection)
        self.x_list.setMaximumHeight(110)
        self.x_list.currentRowChanged.connect(self._x_changed)
        layout.addWidget(self.x_list)

        # Y axis table  (columns: L checkbox | R checkbox | variable name)
        layout.addWidget(QLabel("Y axis  (L = left axis,  R = right axis):"))
        self.y_table = QTableWidget(0, 3)
        self.y_table.setHorizontalHeaderLabels(["L", "R", "Variable"])
        h = self.y_table.horizontalHeader()
        h.setSectionResizeMode(0, ResizeToContents)
        h.setSectionResizeMode(1, ResizeToContents)
        h.setSectionResizeMode(2, StretchLast)
        self.y_table.setSelectionMode(NoSelection)
        self.y_table.setMaximumHeight(180)
        self.y_table.itemChanged.connect(self._y_changed)
        layout.addWidget(self.y_table)
        self._suppress_y = False 

        btn_grid = QWidget()
        btn_grid_layout = QtWidgets.QGridLayout(btn_grid)
        btn_grid_layout.setSpacing(4)
        settings_buttons = [
            ("Line style",  self.popup_line.show),
            ("Axes",        self.popup_axes.show),
            ("Legend",      self.popup_legend.show),
            ("Grid",        self.popup_grid.show),
            ("Title",       self.popup_title.show),
            ("Multi-plot",  self.popup_multi.show),
            ("Fourier",     self.popup_fourier.show),
            ("Avg / RMS",   self.popup_avgrms.show),
            ("Power",       self.popup_power.show),
        ]
        for i, (label, slot) in enumerate(settings_buttons):
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_grid_layout.addWidget(b, i // 2, i % 2)
        layout.addWidget(btn_grid)

        # Plot button
        self.plot_btn = QPushButton("Plot")
        self.plot_btn.setEnabled(False)
        self.plot_btn.clicked.connect(self._plot)
        layout.addWidget(self.plot_btn)

    #  file loading 
    def _load_in_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GSEIM project file", "", "GSEIM projects (*.in)")
        if not path:
            return
        try:
            _, blocks = parse_in_file(path)
        except GseimInFileError as e:
            QMessageBox.warning(self, "Parse error", str(e)); return

        self.output_blocks = blocks
        self.file_label.setText(path)
        self.output_list.clear()
        self._clear_columns()

        if not blocks:
            QMessageBox.information(self, "No output files",
                "No .dat output blocks found in this project."); return

        for b in blocks:
            self.output_list.addItem(QListWidgetItem(b.dat_filename))

        self.fourier.enabled  = False
        self.avgrms.avg = self.avgrms.rms = False
        self.power.enabled    = False
        self.multiplot.enabled = False

    def _select_output_file(self, row):
        if row < 0 or row >= len(self.output_blocks):
            return
        block = self.output_blocks[row]
        try:
            self.data = load_dat(block.dat_path)
        except (FileNotFoundError, ValueError) as e:
            QMessageBox.warning(self, "Load error", str(e))
            self.data = None; self.plot_btn.setEnabled(False); return

        n_rows, n_col = self.data.shape
        self.labels = column_labels(block.variables, n_col)
        self.shape_label.setText(f"{n_rows} rows × {n_col} columns")

        # One default LineStyling per column
        self.line_styles = [
            LineStyling(label=lbl, color=self.COLOR_SET[i % len(self.COLOR_SET)])
            for i, lbl in enumerate(self.labels)
        ]

        self._populate_columns()
        self.axes.x_label = self.labels[0] if self.labels else ""

    def _clear_columns(self):
        self.x_list.clear()
        self._suppress_y = True
        self.y_table.setRowCount(0)
        self._suppress_y = False
        self.x_col = None
        self.y_left_rows  = []
        self.y_right_rows = []
        self.plot_btn.setEnabled(False)

    def _populate_columns(self):
        self._clear_columns()

        for lbl in self.labels:
            self.x_list.addItem(QListWidgetItem(lbl))
        if self.x_list.count():
            self.x_list.setCurrentRow(0)
            self.x_col = 0

        self._suppress_y = True
        self.y_table.setRowCount(len(self.labels))
        for row, lbl in enumerate(self.labels):
            for col in (0, 1):
                item = QTableWidgetItem()
                item.setFlags(UserCheckable | ItemEnabled)
                item.setCheckState(Unchecked)
                self.y_table.setItem(row, col, item)
            name_item = QTableWidgetItem(lbl)
            name_item.setFlags(ItemEnabled)
            self.y_table.setItem(row, 2, name_item)
        self._suppress_y = False

        self.plot_btn.setEnabled(True)

    #  selection handlers 
    def _x_changed(self, row):
        self.x_col = row if row >= 0 else None

    def _y_changed(self, item):
        """Keep L and R mutually exclusive per row."""
        if self._suppress_y or item.column() not in (0, 1):
            return
        if item.checkState() == Checked:
            other_col = 1 - item.column()
            other = self.y_table.item(item.row(), other_col)
            if other and other.checkState() == Checked:
                self._suppress_y = True
                other.setCheckState(Unchecked)
                self._suppress_y = False

        # Rebuild index lists
        self.y_left_rows  = []
        self.y_right_rows = []
        for r in range(self.y_table.rowCount()):
            l_item = self.y_table.item(r, 0)
            r_item = self.y_table.item(r, 1)
            if l_item and l_item.checkState() == Checked:
                self.y_left_rows.append(r)
            elif r_item and r_item.checkState() == Checked:
                self.y_right_rows.append(r)

    #  derived-feature computations 

    def _run_fourier(self):
        """Compute Fourier harmonics for all selected Y columns, save to files."""
        if self.data is None: return
        f   = self.fourier
        t   = self.data[:, self.x_col]
        all_y = self.y_left_rows + self.y_right_rows
        n_x = len(all_y)
        if n_x == 0: return

        base = str(self.output_blocks[self.output_list.currentRow()].dat_path)
        self.file_fourier = base.replace(".dat", "_fourier.dat")
        self.file_thd     = base.replace(".dat", "_thd.dat")

        x_fourier = [[0.0]*f.n_fourier for _ in range(n_x)]
        thd       = [0.0]*n_x

        for idx, col in enumerate(all_y):
            x_fourier[idx], thd[idx] = fourier_coeff(
                t, self.data[:, col], f.t_start, f.t_end, f.n_fourier)

        with open(self.file_fourier, "w") as fp:
            for i in range(f.n_fourier):
                row = "%3d" % i + "".join(" %11.4E" % x_fourier[j][i] for j in range(n_x))
                fp.write(row + "\n")

        with open(self.file_thd, "w") as fp:
            for v in thd:
                fp.write(" %9.3E\n" % v)

    def _run_avgrms(self):
        """Compute rolling average/RMS for all selected Y columns, save to files."""
        if self.data is None: return
        a   = self.avgrms
        T   = a.period
        t   = self.data[:, self.x_col]
        all_y = self.y_left_rows + self.y_right_rows
        n_x = len(all_y)
        if n_x == 0: return

        xs = [self.data[:, col] for col in all_y]

        t_out = []
        x_avg = [[] for _ in range(n_x)]
        x_rms = [[] for _ in range(n_x)]

        sum_avg = [0.0]*n_x
        sum_rms = [0.0]*n_x
        x_last  = [xs[j][0] for j in range(n_x)]
        t_last  = t[0]
        t0      = t[0]
        t_end   = t0 + T
        eps     = 1e-3 * T
        flag    = False

        for i in range(1, len(t)):
            t1 = t[i]
            x1 = [xs[j][i] for j in range(n_x)]

            if abs(t1 - t_end) < eps:
                if not flag:
                    for j in range(n_x):
                        if a.avg:
                            sum_avg[j] += 0.5*(x_last[j]+x1[j])*(t1-t_last)
                            x_avg[j].append(sum_avg[j]/T); sum_avg[j] = 0.0
                        if a.rms:
                            sum_rms[j] += 0.5*(x_last[j]**2+x1[j]**2)*(t1-t_last)
                            x_rms[j].append(np.sqrt(sum_rms[j]/T)); sum_rms[j] = 0.0
                        x_last[j] = x1[j]
                    t_out.append(t_end); t0 = t_end; t_end = t0+T; t_last = t1; flag = True
            elif t1 > t_end:
                xa = [x_last[j]+((x1[j]-x_last[j])/(t1-t_last))*(t_end-t_last) for j in range(n_x)]
                for j in range(n_x):
                    if a.avg:
                        sum_avg[j] += 0.5*(x_last[j]+xa[j])*(t_end-t_last)
                        x_avg[j].append(sum_avg[j]/T); sum_avg[j] = 0.0
                    if a.rms:
                        sum_rms[j] += 0.5*(x_last[j]**2+xa[j]**2)*(t_end-t_last)
                        x_rms[j].append(np.sqrt(sum_rms[j]/T)); sum_rms[j] = 0.0
                    x_last[j] = xa[j]
                t_out.append(t_end); t_last = t_end; t0 = t_end; t_end = t0+T; flag = False
            else:
                for j in range(n_x):
                    if a.avg: sum_avg[j] += 0.5*(x_last[j]+x1[j])*(t1-t_last)
                    if a.rms: sum_rms[j] += 0.5*(x_last[j]**2+x1[j]**2)*(t1-t_last)
                    x_last[j] = x1[j]
                t_last = t1; flag = False

        base = str(self.output_blocks[self.output_list.currentRow()].dat_path)
        if a.avg:
            self.file_avg = base.replace(".dat", "_avg.dat")
            with open(self.file_avg, "w") as fp:
                for i, tv in enumerate(t_out):
                    row = "%11.4E" % tv + "".join(" %11.4E" % x_avg[j][i] for j in range(n_x))
                    fp.write(row+"\n")
        if a.rms:
            self.file_rms = base.replace(".dat", "_rms.dat")
            with open(self.file_rms, "w") as fp:
                for i, tv in enumerate(t_out):
                    row = "%11.4E" % tv + "".join(" %11.4E" % x_rms[j][i] for j in range(n_x))
                    fp.write(row+"\n")


    def _run_power(self):
        """Compute per-period power quantities, save to a file."""
        if self.data is None: return
        p = self.power
        if p.v_name not in self.labels or p.i_name not in self.labels: return
        T = p.period
        t       = self.data[:, self.x_col]
        voltage = self.data[:, self.labels.index(p.v_name)]
        current = self.data[:, self.labels.index(p.i_name)]
        pwr     = voltage * current

        sum_v = sum_i = sum_p = 0.0
        v_last = voltage[0]; i_last = current[0]; p_last = pwr[0]
        t_last = t[0]; t0 = t[0]; t_end = t0+T; eps = 1e-3*T; flag = False
        x_rms_v=[]; x_rms_i=[]; x_avg_p=[]

        for i in range(1, len(t)):
            t1=t[i]; v1=voltage[i]; i1=current[i]; p1=pwr[i]
            if abs(t1-t_end)<eps:
                if not flag:
                    sum_v += 0.5*(v_last**2+v1**2)*(t1-t_last); x_rms_v.append(np.sqrt(sum_v/T)); sum_v=0.0; v_last=v1
                    sum_i += 0.5*(i_last**2+i1**2)*(t1-t_last); x_rms_i.append(np.sqrt(sum_i/T)); sum_i=0.0; i_last=i1
                    sum_p += 0.5*(p_last+p1)*(t1-t_last);       x_avg_p.append(sum_p/T);           sum_p=0.0; p_last=p1
                    t0=t_end; t_end=t0+T; t_last=t1; flag=True
            elif t1>t_end:
                va=v_last+((v1-v_last)/(t1-t_last))*(t_end-t_last)
                ia=i_last+((i1-i_last)/(t1-t_last))*(t_end-t_last)
                pa=p_last+((p1-p_last)/(t1-t_last))*(t_end-t_last)
                sum_v += 0.5*(v_last**2+va**2)*(t_end-t_last); x_rms_v.append(np.sqrt(sum_v/T)); sum_v=0.0; v_last=va
                sum_i += 0.5*(i_last**2+ia**2)*(t_end-t_last); x_rms_i.append(np.sqrt(sum_i/T)); sum_i=0.0; i_last=ia
                sum_p += 0.5*(p_last+pa)*(t_end-t_last);       x_avg_p.append(sum_p/T);           sum_p=0.0; p_last=pa
                t_last=t_end; t0=t_end; t_end=t0+T; flag=False
            else:
                sum_v+=0.5*(v_last**2+v1**2)*(t1-t_last); v_last=v1
                sum_i+=0.5*(i_last**2+i1**2)*(t1-t_last); i_last=i1
                sum_p+=0.5*(p_last+p1)*(t1-t_last);       p_last=p1
                t_last=t1; flag=False

        base = str(self.output_blocks[self.output_list.currentRow()].dat_path)
        self.file_power = base.replace(".dat","_power.dat")
        with open(self.file_power,"w") as fp:
            for k in range(len(x_rms_v)):
                t_start_k = float(k)*T
                t_end_k   = float(k+1)*T
                p_app = x_rms_v[k]*x_rms_i[k]
                pf    = x_avg_p[k]/p_app if p_app else 0.0
                vals = " %11.4E %11.4E %11.4E %11.4E %11.4E" % (
                    x_rms_v[k], x_rms_i[k], x_avg_p[k], p_app, pf)
                fp.write("%11.4E" % t_start_k + vals + "\n")
                fp.write("%11.4E" % t_end_k   + vals + "\n")

    #  plot button 

    def _plot(self):
        if self.data is None: return
        if self.x_col is None:
            QMessageBox.information(self, "Plot", "Pick an X axis column."); return
        if not self.y_left_rows and not self.y_right_rows:
            QMessageBox.information(self, "Plot", "Pick at least one Y axis column."); return

        x       = self.data[:, self.x_col]
        x_label = self.labels[self.x_col]

        settings = dict(axes=self.axes, grid=self.grid,
                        title=self.title, legend=self.legend)

        #  power mode: special 3-subplot window 
        if self.power.enabled and self.file_power:
            self._plot_power(x, x_label, settings); return

        #  fourier mode: bar chart of harmonics 
        if self.fourier.enabled and self.file_fourier:
            self._plot_fourier(x_label, settings); return

        #  collect series 
        left_series  = [(self.data[:, r], self.line_styles[r]) for r in self.y_left_rows]
        right_series = [(self.data[:, r], self.line_styles[r]) for r in self.y_right_rows]

        # Overlay avg/rms as dashed lines if enabled
        if (self.avgrms.avg or self.avgrms.rms) and self.data is not None:
            self._run_avgrms()
            left_series, right_series = self._inject_avgrms(
                left_series, right_series)

        #  multi-plot: stacked subplots 
        if self.multiplot.enabled:
            all_series = ([(y, s, "L") for y, s in left_series] +
                          [(y, s, "R") for y, s in right_series])
            win = MultiPlotWindow(x, x_label, all_series, settings)
        else:
            win = PlotWindow(x, x_label, left_series, right_series, settings)

        win.show()
        self.open_windows.append(win)

    def _inject_avgrms(self, left_series, right_series):
        """Add avg/rms overlay lines as extra (dashed/dashdot) series."""
        try:
            all_y = self.y_left_rows + self.y_right_rows
            new_left  = list(left_series)
            new_right = list(right_series)
            if self.avgrms.avg and self.file_avg:
                data_avg = np.loadtxt(self.file_avg)
                t_avg = data_avg[:, 0]
                for idx, col in enumerate(all_y):
                    y_avg = data_avg[:, idx+1]
                    s = LineStyling(
                        label      = self.labels[col]+"_avg",
                        line_style = "dashed",
                        color      = self.line_styles[col].color,
                        width      = 1.0)
                    if col in self.y_left_rows:
                        new_left.append((y_avg, s))
                    else:
                        new_right.append((y_avg, s))
            if self.avgrms.rms and self.file_rms:
                data_rms = np.loadtxt(self.file_rms)
                t_rms = data_rms[:, 0]
                for idx, col in enumerate(all_y):
                    y_rms = data_rms[:, idx+1]
                    s = LineStyling(
                        label      = self.labels[col]+"_rms",
                        line_style = "dashdot",
                        color      = self.line_styles[col].color,
                        width      = 1.0)
                    if col in self.y_left_rows:
                        new_left.append((y_rms, s))
                    else:
                        new_right.append((y_rms, s))
            return new_left, new_right
        except Exception:
            return left_series, right_series

    def _plot_fourier(self, x_label, settings):
        """Open a window showing Fourier harmonics as bar charts."""
        data_f = np.loadtxt(self.file_fourier)
        thd = np.atleast_1d(np.loadtxt(self.file_thd))
        all_y  = self.y_left_rows + self.y_right_rows
        n      = len(all_y)

        win = QMainWindow()
        win.setWindowTitle("Fourier Harmonics")
        fig = Figure(figsize=(7, 3*n), tight_layout=True)
        canvas = FigureCanvas(fig)
        for idx, col in enumerate(all_y):
            ax = fig.add_subplot(n, 1, idx+1)
            ax.bar(data_f[:, 0], data_f[:, idx+1], width=0.5,
                   color=self.line_styles[col].color)
            ax.set_xlabel("Harmonic index")
            ax.set_ylabel(self.labels[col])
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            thd_val = thd[idx]
            ax.set_title(f"{self.labels[col]}  —  THD = {thd_val:.4f}")
            ax.grid(True, color="lightgrey")
        w = QWidget(); l = QVBoxLayout(w); l.addWidget(canvas)
        win.setCentralWidget(w)
        win.addToolBar(TrimmedToolbar(canvas, win))
        win.resize(720, 300*n)
        win.show()
        self.open_windows.append(win)

    def _plot_power(self, x, x_label, settings):
        """Open a 3-subplot power window (V&I, P_avg & P_app, power factor)."""
        p = self.power
        data_pw = np.loadtxt(self.file_power)
        t_pw    = data_pw[:, 0]
        v_rms   = data_pw[:, 1]; i_rms = data_pw[:, 2]
        p_avg   = data_pw[:, 3]; p_app = data_pw[:, 4]; pf = data_pw[:, 5]
        v_idx   = self.labels.index(p.v_name)
        i_idx   = self.labels.index(p.i_name)
        voltage = self.data[:, v_idx]
        current = self.data[:, i_idx]

        win = QMainWindow()
        win.setWindowTitle("Power Analysis")
        fig = Figure(figsize=(7, 9), tight_layout=True)
        canvas = FigureCanvas(fig)
        fig.subplots_adjust(hspace=0.05)

        ax1 = fig.add_subplot(3, 1, 1)
        ax1.plot(x, voltage, color="dodgerblue",  label=p.v_name)
        ax1.plot(t_pw, v_rms, color="dodgerblue", linestyle="dashed", label=p.v_name+"_rms")
        ax1r = ax1.twinx()
        ax1r.plot(x, current, color="orangered",  label=p.i_name)
        ax1r.plot(t_pw, i_rms, color="orangered", linestyle="dashed", label=p.i_name+"_rms")
        lines = ax1.get_lines() + ax1r.get_lines()
        ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
        ax1.set_ylabel("V  /  I"); ax1.grid(color="lightgrey"); ax1.label_outer()

        ax2 = fig.add_subplot(3, 1, 2)
        ax2.plot(t_pw, p_avg, color="teal",     label="P_avg")
        ax2.plot(t_pw, p_app, color="deeppink", label="P_app")
        ax2.legend(loc="best", fontsize=8)
        ax2.set_ylabel("Power"); ax2.grid(color="lightgrey"); ax2.label_outer()

        ax3 = fig.add_subplot(3, 1, 3)
        ax3.plot(t_pw, pf, color="darkgoldenrod", label="Power Factor")
        ax3.legend(loc="best", fontsize=8)
        ax3.set_xlabel(x_label); ax3.set_ylabel("PF"); ax3.grid(color="lightgrey")

        w = QWidget(); l = QVBoxLayout(w); l.addWidget(canvas)
        win.setCentralWidget(w)
        win.addToolBar(TrimmedToolbar(canvas, win))
        win.resize(720, 700)
        win.show()
        self.open_windows.append(win)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

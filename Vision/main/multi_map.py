# map.py
import sys
import math
from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPainter, QColor, QPainterPath
from PySide6.QtCore import Qt

_app = None
_win = None

class _Map(QWidget):
    def __init__(self):
        super().__init__()
        self.points = []          # [(x, y, dist), ...]
        self.heading = 0.0        # radians (0 = +x, CCW positive)
        self.scale = 15           # pixels par unité
        self.fov_deg = 60         # champ de vision
        self.fov_range = 20.0      # portée du cône en "unités monde"
        self.resize(600, 400)
        self.setWindowTitle("Map")

    def _world_to_screen(self, x, y):
        w, h = self.width(), self.height()
        sx = w / 2 + x * self.scale
        sy = h / 2 - y * self.scale
        return sx, sy

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.fillRect(self.rect(), Qt.white)

        # Origine (0,0) au centre
        ox, oy = self._world_to_screen(0.0, 0.0)

        # --- Cône FOV semi-transparent ---
        radius_px = self.fov_range * self.scale
        rect = (ox - radius_px, oy - radius_px, 2 * radius_px, 2 * radius_px)

        heading_deg = math.degrees(self.heading)
        # Conversion monde -> écran (y inversé)
        center_deg_screen = -heading_deg
        start_deg = center_deg_screen - self.fov_deg / 2.0
        span_deg = self.fov_deg

        path = QPainterPath()
        path.moveTo(ox, oy)
        # arcTo: angles en degrés, CCW, 0° = +x
        path.arcTo(*rect, start_deg, span_deg)
        path.closeSubpath()

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 120, 255, 80))  # bleu semi-transparent
        p.drawPath(path)

        # Petit point à l'origine
        p.setBrush(QColor(0, 0, 0))
        p.drawEllipse(int(ox - 3), int(oy - 3), 6, 6)

        # --- Points + distance ---
        for x, y, d, color in self.points:
            if color == "black":
                p.setBrush(QColor(0, 0, 0))
                p.setPen(Qt.black)
            else:
                p.setBrush(QColor(160, 160, 160))
                p.setPen(Qt.black)

            sx, sy = self._world_to_screen(x, y)
            p.drawEllipse(int(sx - 3), int(sy - 3), 6, 6)
            p.drawText(int(sx + 6), int(sy - 6), f"{d:.1f}m")




def show_map(points, heading):
    """
    points: [(x, y, dist), ...]
    heading: radians (0 = vers +x, CCW positif)
    """
    global _app, _win

    if _app is None:
        _app = QApplication.instance()
        if _app is None:
            _app = QApplication(sys.argv)

    if _win is None:
        _win = _Map()
        _win.show()

    _win.points = points
    _win.heading = float(heading)
    _win.update()

    # Non-bloquant, compatible avec ta boucle while
    _app.processEvents()

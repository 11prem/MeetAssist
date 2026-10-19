"""Design tokens + theme engine (Apple-style minimalism, light & dark).

Three layers: primitives (the Palette values) -> semantic roles (bg, text, accent...) -> component styles (build_qss).
Every colour in the UI comes from the active Palette; nothing is hard-coded in widgets.
Contrast (WCAG): text on bg >= 7:1, text2 >= 7:1, text3 >= 4.5:1, accent text >= 4.5:1 in BOTH themes.
"""
from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontDatabase, QGuiApplication, QPalette

from .. import paths

# ----------------------------------------------------------------------------- tokens: spacing / radius / type / motion
SP = {"xxs": 2, "xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24, "xxl": 32, "huge": 48}      # 4-pt rhythm
RADIUS = {"sm": 6, "md": 10, "lg": 14, "xl": 20}
ICON = {"sm": 14, "md": 18, "lg": 22, "xl": 28}                                              # one size scale for icons
MOTION = {"fast": 120, "base": 200, "slow": 320, "pulse": 1700}                              # ms
FONT_STACK = "'Fraunces','Segoe UI Variable Text','Segoe UI','Nirmala UI'"
TYPE = {  # px (device independent)
    "caption": 12, "footnote": 13, "body": 14, "callout": 15, "headline": 17, "title": 26, "timer": 22, "hero": 30,
}
READ_WIDTH = 760                                                                              # transcript measure (px)


@dataclass(frozen=True)
class Palette:
    name: str
    dark: bool
    bg: str            # content background
    sidebar: str       # sidebar background
    surface: str       # raised surfaces (cards, dock, menus)
    fill: str          # quiet control fills (search field, segmented track, icon-button hover)
    fill_hover: str
    fill_press: str
    text: str
    text2: str
    text3: str
    placeholder: str   # decorative / placeholder only (3:1)
    sep: str           # hairlines
    border: str        # control borders
    accent: str
    accent_hover: str
    accent_press: str
    accent_tint: str   # selected rows, soft accent backgrounds
    accent_fill: str   # filled controls with on_accent text (>= 4.5:1 in both themes)
    accent_fill_hover: str
    accent_fill_press: str
    on_accent: str
    record: str
    record_hover: str
    record_tint: str
    success: str
    success_tint: str
    warning: str
    warning_tint: str
    you: str           # speaker colours
    remote: str
    shadow: str        # rgba for floating surfaces


LIGHT = Palette(
    name="light", dark=False,
    bg="#FFFFFF", sidebar="#F5F5F7", surface="#FFFFFF", fill="#EEEEF1", fill_hover="#E6E6EA", fill_press="#DCDCE1",
    text="#1D1D1F", text2="#515154", text3="#636366", placeholder="#8E8E93", sep="#E5E5EA", border="#D2D2D7",
    accent="#0071E3", accent_hover="#0077ED", accent_press="#0062C3", accent_tint="#E3EFFD", accent_fill="#0071E3", accent_fill_hover="#0077ED", accent_fill_press="#0062C3", on_accent="#FFFFFF",
    record="#FF3B30", record_hover="#E5352B", record_tint="#FDECEA",
    success="#1A7F37", success_tint="#E6F4EA", warning="#9A4A00", warning_tint="#FFF4E0",
    you="#0071E3", remote="#1A7F37", shadow="rgba(0,0,0,0.14)",
)
DARK = Palette(
    name="dark", dark=True,
    bg="#0B0C10", sidebar="#101218", surface="#16181F", fill="#20232C", fill_hover="#2A2E38", fill_press="#333743",
    text="#F5F5F7", text2="#C7C7CC", text3="#A8A8AE", placeholder="#6E6E73", sep="#1E2128", border="#363A44",
    accent="#2997FF", accent_hover="#4AA8FF", accent_press="#1A84E8", accent_tint="#16324F", accent_fill="#0A6CD6", accent_fill_hover="#1479E8", accent_fill_press="#085CB8", on_accent="#FFFFFF",
    record="#FF453A", record_hover="#FF6259", record_tint="#4A2220",
    success="#30D158", success_tint="#173323", warning="#FFB340", warning_tint="#3F2F10",
    you="#A79CFF", remote="#5FE0D3", shadow="rgba(0,0,0,0.55)",
)


def rgba_str(hex_color: str, alpha: float) -> str:
    """A literal CSS rgba() string for QSS (Qt's stylesheet engine parses this directly;
    QColor(str) does not, so this never feeds contrast() or qcolor())."""
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


def contrast(fg: str, bg: str) -> float:
    def lum(c: str) -> float:
        q = QColor(c)
        ch = [q.redF(), q.greenF(), q.blueF()]
        ch = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


# ----------------------------------------------------------------------------- theme manager
class ThemeManager(QObject):
    changed = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.mode = "system"                       # system | light | dark
        self._p = LIGHT
        hints = QGuiApplication.styleHints()
        try:
            hints.colorSchemeChanged.connect(lambda *_: self.refresh())
        except Exception:
            pass
        self.refresh()

    @property
    def p(self) -> Palette:
        return self._p

    def set_mode(self, mode: str) -> None:
        self.mode = mode if mode in ("system", "light", "dark") else "system"
        self.refresh()

    def refresh(self) -> None:
        if self.mode == "light":
            dark = False
        elif self.mode == "dark":
            dark = True
        else:
            try:
                dark = QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
            except Exception:
                dark = False
        new = DARK if dark else LIGHT
        if new is not self._p or True:
            self._p = new
            self._apply_app_palette()
            self.changed.emit()

    def _apply_app_palette(self) -> None:
        app = QGuiApplication.instance()
        if app is None:
            return
        p, pal = self._p, QPalette()
        role = QPalette.ColorRole
        for r, c in ((role.Window, p.bg), (role.WindowText, p.text), (role.Base, p.surface), (role.AlternateBase, p.fill),
                     (role.Text, p.text), (role.Button, p.surface), (role.ButtonText, p.text), (role.ToolTipBase, p.text),
                     (role.ToolTipText, p.bg), (role.Highlight, p.accent), (role.HighlightedText, p.on_accent),
                     (role.PlaceholderText, p.placeholder), (role.Link, p.accent)):
            pal.setColor(r, QColor(c))
        pal.setColor(QPalette.ColorGroup.Disabled, role.Text, QColor(p.placeholder))
        pal.setColor(QPalette.ColorGroup.Disabled, role.ButtonText, QColor(p.placeholder))
        app.setPalette(pal)


theme: ThemeManager | None = None
_fonts_loaded = False


def load_bundled_fonts() -> None:
    """Load fonts shipped in assets/fonts/ so the look never depends on what happens to be installed
    system-wide - no admin rights needed (see AGENTS.md). Safe to call more than once."""
    global _fonts_loaded
    if _fonts_loaded:
        return
    _fonts_loaded = True
    f = paths.APP_ROOT / "assets" / "fonts" / "fraunces" / "Fraunces-Variable.ttf"
    if f.exists():
        QFontDatabase.addApplicationFont(str(f))


def init() -> ThemeManager:
    global theme
    if theme is None:
        load_bundled_fonts()
        theme = ThemeManager()
    return theme


def P() -> Palette:
    return init().p


def app_font() -> QFont:
    f = QFont("Fraunces")
    f.setFamilies(["Fraunces", "Segoe UI Variable Text", "Segoe UI", "Nirmala UI"])
    f.setPixelSize(TYPE["body"])
    f.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    return f


def tabular(font: QFont) -> QFont:
    """Equal-width digits so a running timer does not jitter."""
    try:
        font.setFeature("tnum", 1)
    except Exception:
        pass
    return font


def style_titlebar(window, p: Palette) -> None:
    """Windows 11: dark/light caption + caption colour = content background (seamless, Apple-like)."""
    if sys.platform != "win32":
        return
    try:
        hwnd = int(window.winId())
        dwm = ctypes.windll.dwmapi

        def setattr_(attr: int, val: int) -> None:
            v = ctypes.c_int(val)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))

        def colorref(hexcol: str) -> int:
            c = QColor(hexcol)
            return c.red() | (c.green() << 8) | (c.blue() << 16)

        setattr_(20, 1 if p.dark else 0)            # DWMWA_USE_IMMERSIVE_DARK_MODE
        setattr_(33, 2)                             # DWMWA_WINDOW_CORNER_PREFERENCE = ROUND (Win11): frameless windows default to square
        setattr_(35, colorref(p.bg))                # DWMWA_CAPTION_COLOR (Win11 22H2+) - matches the window-edge gutter, not the floating sidebar card
        setattr_(36, colorref(p.text))              # DWMWA_TEXT_COLOR
        setattr_(34, colorref(p.sep))               # DWMWA_BORDER_COLOR
    except Exception:
        pass


# ----------------------------------------------------------------------------- component styles
def build_qss(p: Palette) -> str:
    r, t = RADIUS, TYPE
    nebula = (f"qradialgradient(cx:0.15, cy:0.08, radius:1.3, fx:0.15, fy:0.08, "
              f"stop:0 {rgba_str(p.you, 0.20)}, stop:0.35 {rgba_str(p.you, 0.04)}, "
              f"stop:0.65 {rgba_str(p.remote, 0.05)}, stop:1 {p.bg})")
    mainbg = nebula if p.dark else p.bg
    # Dark mode only: sidebar/page become floating glass cards over #mainbg's nebula, with a
    # visible gutter (see MainWindow._build's root margins) instead of flat edge-to-edge regions.
    glass_panels = (f"QWidget#content {{ background: {rgba_str(p.bg, 0.0)}; }}\n"
                    f"QWidget#sidebar {{ background: {rgba_str(p.sidebar, 0.55)}; border: 1px solid {rgba_str(p.border, 0.4)}; border-radius: {r['lg']}px; }}\n"
                    f"QWidget#page {{ background: {rgba_str(p.surface, 0.35)}; border: 1px solid {rgba_str(p.border, 0.35)}; border-radius: {r['lg']}px; }}\n"
                    f"QFrame#vsep {{ background: transparent; }}\n"          # the gutter itself separates sidebar/content now
                    if p.dark else "")
    return f"""
* {{ font-family: {FONT_STACK}; font-size: {t['body']}px; color: {p.text}; outline: 0; }}
QMainWindow, QDialog {{ background: {p.bg}; }}
QWidget#mainbg {{ background: {mainbg}; }}
QWidget#titlebar {{ background: transparent; }}
QWidget#content {{ background: {p.bg}; }}
QWidget#page {{ background: {p.bg}; border-radius: {r['lg']}px; }}
QWidget#sidebar {{ background: {p.sidebar}; border-radius: {r['lg']}px; }}
{glass_panels}
QFrame#vsep {{ background: {p.sep}; border: none; max-width: 1px; min-width: 1px; }}
QFrame#hsep {{ background: {p.sep}; border: none; max-height: 1px; min-height: 1px; }}
QToolTip {{ background: transparent; color: {p.text2}; border: none; padding: 2px; font-size: {t['footnote']}px; }}

QLabel {{ background: transparent; }}
QLabel[role="hero"]     {{ font-size: {t['hero']}px; font-weight: 700; }}
QLabel[role="title"]    {{ font-size: {t['title']}px; font-weight: 700; letter-spacing: -0.3px; }}
QLabel[role="headline"] {{ font-size: {t['headline']}px; font-weight: 600; }}
QLabel[role="callout"]  {{ font-size: {t['callout']}px; font-weight: 600; }}
QLabel[role="body2"]    {{ color: {p.text2}; font-size: {t['body']}px; }}
QLabel[role="footnote"] {{ color: {p.text3}; font-size: {t['footnote']}px; }}
QLabel[role="caption"]  {{ color: {p.text3}; font-size: {t['caption']}px; }}
QLabel[role="section"]  {{ color: {p.text3}; font-size: {t['caption']}px; font-weight: 600; }}

QLineEdit {{ background: {p.fill}; border: 1px solid transparent; border-radius: {r['md']}px; padding: 7px 10px;
            selection-background-color: {p.accent}; selection-color: {p.on_accent}; }}
QLineEdit:hover {{ background: {p.fill_hover}; }}
QLineEdit:focus {{ background: {p.surface}; border: 2px solid {p.accent}; padding: 6px 9px; }}
QLineEdit[flat="true"] {{ background: transparent; border: none; padding: 0; font-size: {t['callout']}px; font-weight: 600; }}
QLineEdit[flat="true"]:focus {{ background: transparent; border: none; padding: 0; }}
QPlainTextEdit {{ background: {p.fill}; border: 1px solid transparent; border-radius: {r['md']}px; padding: 8px 10px;
                 selection-background-color: {p.accent}; selection-color: {p.on_accent}; }}
QPlainTextEdit:focus {{ background: {p.surface}; border: 2px solid {p.accent}; padding: 7px 9px; }}

QPushButton {{ background: {p.fill}; border: none; border-radius: {r['md']}px; padding: 8px 16px; font-weight: 500; }}
QPushButton:hover {{ background: {p.fill_hover}; }}
QPushButton:pressed {{ background: {p.fill_press}; }}
QPushButton:focus {{ border: 2px solid {p.accent}; padding: 6px 14px; }}
QPushButton:disabled {{ color: {p.placeholder}; background: {p.fill}; }}
QPushButton[variant="primary"] {{ background: {p.accent_fill}; color: {p.on_accent}; font-weight: 600; }}
QPushButton[variant="primary"]:hover {{ background: {p.accent_fill_hover}; }}
QPushButton[variant="primary"]:pressed {{ background: {p.accent_fill_press}; }}
QPushButton[variant="primary"]:disabled {{ background: {p.fill}; color: {p.placeholder}; }}
QPushButton[variant="record"] {{ background: {p.record}; color: #FFFFFF; font-weight: 600; }}
QPushButton[variant="record"]:hover {{ background: {p.record_hover}; }}
QPushButton[variant="plain"] {{ background: transparent; color: {p.accent}; padding: 6px 8px; font-weight: 500; }}
QPushButton[variant="plain"]:hover {{ background: {p.accent_tint}; }}
QPushButton[variant="side"] {{ background: transparent; text-align: left; padding: 8px 10px; border-radius: {r['md']}px; color: {p.text2}; font-weight: 500; }}
QPushButton[variant="side"]:hover {{ background: {p.fill}; color: {p.text}; }}
QPushButton[variant="side"]:focus {{ border: 2px solid {p.accent}; padding: 6px 8px; }}

QToolButton {{ background: transparent; border: none; border-radius: {r['md']}px; padding: 6px; color: {p.text2}; }}
QToolButton:hover {{ background: {p.fill}; }}
QToolButton:pressed {{ background: {p.fill_press}; }}
QToolButton:focus {{ border: 2px solid {p.accent}; padding: 4px; }}
QToolButton:disabled {{ color: {p.placeholder}; }}
QToolButton[labelled="true"] {{ padding: 6px 10px; font-weight: 500; }}
QToolButton[labelled="true"]:focus {{ padding: 4px 8px; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}

QMenu {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {r['md']}px; padding: 6px; }}
QMenu::item {{ padding: 7px 30px 7px 12px; border-radius: {r['sm']}px; color: {p.text}; }}
QMenu::item:selected {{ background: {p.accent_fill}; color: {p.on_accent}; }}
QMenu::item:disabled {{ color: {p.placeholder}; }}
QMenu::separator {{ height: 1px; background: {p.sep}; margin: 6px 8px; }}
QMenu::indicator {{ width: 14px; height: 14px; left: 8px; }}

QScrollBar:vertical {{ background: transparent; width: 12px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.border}; border-radius: 4px; min-height: 36px; margin: 0 2px; }}
QScrollBar::handle:vertical:hover {{ background: {p.placeholder}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; height: 0; width: 0; }}
QScrollBar:horizontal {{ height: 0; }}

QListWidget {{ background: transparent; border: none; outline: 0; }}
QTextBrowser {{ background: transparent; border: none; selection-background-color: {p.accent_tint}; selection-color: {p.text}; }}
QStatusBar {{ background: {p.bg}; color: {p.text3}; border-top: 1px solid {p.sep}; }}
QStatusBar::item {{ border: none; }}

QProgressBar {{ background: {p.fill}; border: none; border-radius: 2px; max-height: 4px; min-height: 4px; }}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 2px; }}

QComboBox {{ background: {p.fill}; border: 1px solid transparent; border-radius: {r['md']}px; padding: 7px 12px; min-height: 20px; }}
QComboBox:hover {{ background: {p.fill_hover}; }}
QComboBox:focus {{ border: 2px solid {p.accent}; padding: 6px 11px; }}
QComboBox::drop-down {{ border: none; width: 26px; }}
QComboBox QAbstractItemView {{ background: {p.surface}; border: 1px solid {p.border}; border-radius: {r['md']}px; padding: 4px;
                              selection-background-color: {p.accent_fill}; selection-color: {p.on_accent}; outline: 0; }}

QMessageBox {{ background: {p.surface}; }}
QDialog#settings {{ background: {p.bg}; }}
QListWidget#nav {{ background: {p.sidebar}; border-right: 1px solid {p.sep}; padding: 12px 8px; }}
QListWidget#nav::item {{ padding: 9px 12px; border-radius: {r['md']}px; color: {p.text2}; margin: 1px 0; }}
QListWidget#nav::item:selected {{ background: {p.accent_fill}; color: {p.on_accent}; }}
QListWidget#nav::item:hover:!selected {{ background: {p.fill}; }}
QFrame#card {{ background: {rgba_str(p.surface, 0.55) if p.dark else p.surface}; border: 1px solid {rgba_str(p.border, 0.5) if p.dark else p.sep}; border-radius: {r['lg']}px; }}
QFrame#banner {{ border-radius: {r['md']}px; }}
QFrame#dock {{ background: {rgba_str(p.surface, 0.5) if p.dark else p.surface}; border: 1px solid {rgba_str(p.border, 0.55) if p.dark else p.sep}; border-radius: {r['xl']}px; }}
QFrame#actionbar {{ border: none; border-top: 1px solid {p.sep}; background: transparent; }}
QFrame#composer {{ background: {rgba_str(p.fill, 0.8) if p.dark else p.fill}; border: 1px solid {rgba_str(p.border, 0.4) if p.dark else "transparent"}; border-radius: 22px; }}
QFrame#composer QLineEdit[flat="true"] {{ font-size: {t['body']}px; }}
QScrollArea {{ background: transparent; border: none; }}
"""

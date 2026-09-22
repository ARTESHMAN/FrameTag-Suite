from pathlib import Path
import sys
from PyQt6.QtWidgets import QApplication
from gui.main_window import DarkLabelMainWindow

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Critical for custom dark styling on Linux

    style_path = Path(__file__).resolve().parent / "assets" / "style.qss"
    stylesheet = ""
    if style_path.exists():
        with open(style_path, "r", encoding="utf-8") as f:
            stylesheet = f.read()
        app.setStyleSheet(stylesheet)
    else:
        print(f"Warning: stylesheet not found at {style_path}")

    window = DarkLabelMainWindow()

    # Apply directly to window to override any internal defaults
    if stylesheet:
        window.setStyleSheet(stylesheet)

    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
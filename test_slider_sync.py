import sys
from PyQt5.QtWidgets import QApplication
from app import OptionsSimApp

def test_slider_and_iv_sync():
    app = QApplication(sys.argv)
    window = OptionsSimApp()
    
    # 1. Test initial default: slider at -0.08, txt_skew at -0.08
    print("Initial txt_skew:", window.txt_skew.text())
    print("Initial slider value:", window.slider_skew.value())
    assert float(window.txt_skew.text()) == -0.08
    assert window.slider_skew.value() == -80
    print("Initial IV Preview Text:", window.lbl_iv_preview.text())
    assert "Est. IV at Upside:" in window.lbl_iv_preview.text()

    # 2. Test typing into txt_skew -> should slide slider and update IV preview text
    window.txt_skew.setText("-0.25")
    print("After setting txt_skew to -0.25, slider is:", window.slider_skew.value())
    assert window.slider_skew.value() == -250
    print("Updated IV Preview Text:", window.lbl_iv_preview.text())
    print("Updated ΔIV Text:", window.lbl_iv_deltas.text())
    assert "Est. IV at Upside:" in window.lbl_iv_preview.text()

    # 3. Test moving slider -> should update txt_skew and IV preview text
    window.slider_skew.setValue(100) # +0.10
    print("After setting slider to 100, txt_skew is:", window.txt_skew.text())
    assert abs(float(window.txt_skew.text()) - 0.10) < 1e-4
    print("IV Preview Text with positive skew:", window.lbl_iv_preview.text())

    # 4. Test changing SF, Upside, or IV text -> should update IV preview text
    window.txt_upside.setText("8400.0")
    print("After changing upside target to 8400.0, IV preview:", window.lbl_iv_preview.text())
    assert "Est. IV at Upside:" in window.lbl_iv_preview.text()

    print("All slider, text sync, and dynamic IV preview tests PASSED!")

if __name__ == '__main__':
    test_slider_and_iv_sync()

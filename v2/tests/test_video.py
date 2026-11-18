import numpy as np

from rollcall.ui.video import fit, to_ppm


def test_ppm_header_and_rgb_order():
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = (255, 0, 0)  # pure blue in OpenCV's BGR order
    data = to_ppm(frame)
    header = b"P6 3 2 255 "
    assert data.startswith(header)
    assert len(data) == len(header) + 2 * 3 * 3
    assert data[len(header):len(header) + 3] == bytes([0, 0, 255])  # RGB: blue is last


def test_fit_keeps_the_shape():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    resized, scale = fit(frame, 320, 320)
    assert resized.shape[:2] == (240, 320) and scale == 0.5
    same, scale = fit(frame, 640, 480)
    assert same is frame and scale == 1.0

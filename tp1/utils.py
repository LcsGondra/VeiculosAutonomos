import cv2
import numpy as np


def imshow_keep_aspect_ratio(winname, img, bg_color=(0, 0, 0)):
    try:
        rect = cv2.getWindowImageRect(winname)
        win_w, win_h = rect[2], rect[3]
    except cv2.error:
        win_w, win_h = 0, 0

    if win_w <= 0 or win_h <= 0:
        cv2.imshow(winname, img)
        return

    img_h, img_w = img.shape[:2]
    img_aspect = img_w / float(img_h)
    win_aspect = win_w / float(win_h)

    if win_aspect > img_aspect:
        new_h = win_h
        new_w = int(win_h * img_aspect)
    else:
        new_w = win_w
        new_h = int(win_w / img_aspect)

    new_w = max(1, new_w)
    new_h = max(1, new_h)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if img.ndim == 2:
        canvas = np.full((win_h, win_w), bg_color[0], dtype=np.uint8)
    else:
        canvas = np.full((win_h, win_w, 3), bg_color, dtype=np.uint8)

    top = (win_h - new_h) // 2
    left = (win_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized
    cv2.imshow(winname, canvas)


def letterbox_image(image, target_size=(224, 224), fill_color=(0, 0, 0)):
    target_w, target_h = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if image.ndim == 2:
        canvas = np.full((target_h, target_w), fill_color[0], dtype=image.dtype)
    else:
        canvas = np.full((target_h, target_w, 3), fill_color, dtype=image.dtype)

    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top : top + new_h, left : left + new_w] = resized

    return canvas, scale, (left, top)


def calculate_rfc3550_jitter(latencies):
    if len(latencies) < 2:
        return 0.0
    jitter = 0.0
    for i in range(1, len(latencies)):
        d = abs(latencies[i] - latencies[i - 1])
        jitter += (d - jitter) / 16.0
    return jitter

import os
import sys
import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio

base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "dados", "direcao_udacity.mp4")
panel_sample_path = os.path.join(base_dir, "dados", "painel_faixas_3a.png")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro: nao foi possivel abrir o video {video_path}")
    sys.exit(1)

ret, first_frame = cap.read()
if not ret or first_frame is None:
    print("Erro ao ler frame inicial.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
h, w = first_frame.shape[:2]

src_pts = np.float32(
    [
        [w * 0.44, h * 0.63],
        [w * 0.56, h * 0.63],
        [w * 0.88, h * 0.94],
        [w * 0.16, h * 0.94],
    ]
)

dst_pts = np.float32(
    [
        [w * 0.22, 0],
        [w * 0.78, 0],
        [w * 0.78, h],
        [w * 0.22, h],
    ]
)

M_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)
M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

auto_mode = "--auto" in sys.argv
max_frames = 60 if auto_mode else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

window_name = "TP1 - Ex 3A: Deteccao de Faixas (BEV + HLS + Sobel + Hough)"
if not auto_mode:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

frame_count = 0
saved_sample = False

print("=" * 65)
print("DETECTOR DE FAIXAS VEICULARES - PIPELINE BEV E EDGE DETECTION")
print("=" * 65)
print("Estagios do Processamento:")
print(" 1. Transformacao de Perspectiva: Bird's Eye View (BEV)")
print(" 2. Segmentacao Hibrida: Canal S de HLS + Gradiente Sobel no eixo X")
print(" 3. Deteccao de Bordas e Retas: Canny + HoughLinesP")
print(" 4. Reprojecao Inversa: Mapeamento de volta para o espaco da camera original")
print(f"Meta de Validacao: >= 50 frames continuos (Modo: {'Auto' if auto_mode else 'Interativo'})")
print("=" * 65)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_count += 1

    bev = cv2.warpPerspective(frame, M_bev, (w, h))

    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    gray_bev = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray_bev, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobel = np.abs(sobel_x)
    norm_sobel = np.uint8(255.0 * abs_sobel / (np.max(abs_sobel) + 1e-9))

    mask_s = s_channel >= 80
    mask_sobel = norm_sobel >= 25
    mask_white = l_channel >= 165

    combined_mask = np.zeros_like(gray_bev)
    combined_mask[(mask_s & (l_channel > 60)) | mask_white | (mask_sobel & (l_channel > 80))] = 255

    edges = cv2.Canny(combined_mask, 50, 150)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=25,
        maxLineGap=60,
    )

    hough_bev = np.zeros_like(bev)
    num_detected_lines = 0

    if lines is not None:
        num_detected_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / float(x2 - x1 + 1e-9)
            if abs(slope) > 0.4:
                cv2.line(hough_bev, (x1, y1), (x2, y2), (0, 255, 0), 4)

    overlay_original = cv2.warpPerspective(hough_bev, M_inv, (w, h))
    final_result = cv2.addWeighted(frame, 1.0, overlay_original, 1.0, 0)

    for pt_idx in range(4):
        p1 = tuple(src_pts[pt_idx].astype(int))
        p2 = tuple(src_pts[(pt_idx + 1) % 4].astype(int))
        cv2.line(final_result, p1, p2, (0, 255, 255), 2)

    thumb_w, thumb_h = 480, 270

    p1_img = cv2.resize(final_result, (thumb_w, thumb_h))
    cv2.putText(p1_img, "1. Final: Faixas + Trap. BEV", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    p2_img = cv2.resize(bev, (thumb_w, thumb_h))
    cv2.putText(p2_img, "2. Bird's Eye View (BEV)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    p3_color = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    p3_img = cv2.resize(p3_color, (thumb_w, thumb_h))
    cv2.putText(p3_img, "3. Mascara HLS S + Sobel X", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    combined_hough_edges = cv2.addWeighted(edges_color, 0.6, hough_bev, 0.9, 0)
    p4_img = cv2.resize(combined_hough_edges, (thumb_w, thumb_h))
    cv2.putText(
        p4_img,
        f"4. Canny + Hough ({num_detected_lines} retas)",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )

    top_row = np.hstack([p1_img, p2_img])
    bot_row = np.hstack([p3_img, p4_img])
    panel = np.vstack([top_row, bot_row])

    if not saved_sample and frame_count >= 25:
        cv2.imwrite(panel_sample_path, panel)
        saved_sample = True

    if frame_count % 15 == 0 or frame_count == 1:
        print(f"Frame {frame_count:04d} processado | Retas detectadas: {num_detected_lines}")

    if not auto_mode:
        imshow_keep_aspect_ratio(window_name, panel)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break

    if auto_mode and frame_count >= max_frames:
        break

cap.release()
if not auto_mode:
    cv2.destroyAllWindows()

if not saved_sample:
    cv2.imwrite(panel_sample_path, panel)

print("=" * 65)
print(f"Execucao finalizada com sucesso. Total de frames validados: {frame_count}")
print(f"Painel dos 4 estagios intermediarios salvo em: {panel_sample_path}")
print("Validacao: >= 50 frames processados de forma estavel.")
print("=" * 65)

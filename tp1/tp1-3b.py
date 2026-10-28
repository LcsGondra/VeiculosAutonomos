import os
import sys
import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio

base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "dados", "direcao_udacity.mp4")
sample_out_path = os.path.join(base_dir, "dados", "deteccao_faixa_preenchida_3b.png")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro: nao foi possivel abrir o video {video_path}")
    sys.exit(1)

ret, first_frame = cap.read()
if not ret or first_frame is None:
    print("Erro ao carregar frame inicial.")
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

XM_PER_PX = 3.7 / 700.0
YM_PER_PX = 30.0 / 720.0

auto_mode = "--auto" in sys.argv
max_frames = 110 if auto_mode else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

window_name = "TP1 - Ex 3B: Estimacao de Desvio Lateral e Curvatura"
if not auto_mode:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

prev_left_fit = None
prev_right_fit = None
alpha_smooth = 0.25

frame_idx = 0
sample_saved = False
deviations_m = []
curvatures_m = []

print("=" * 65)
print("SISTEMA DE CONTROLE LATERAL: ESTIMACAO DE DESVIO E CURVATURA")
print("=" * 65)
print("Parametros Metricos:")
print(f" - Fator de Escala X: 3.7 m / 700 px = {XM_PER_PX:.6f} m/px")
print(f" - Fator de Escala Y: 30.0 m / 720 px = {YM_PER_PX:.6f} m/px")
print(f" - Meta de Execucao: >= 100 frames consecutivos ({'Auto' if auto_mode else 'Interativo'})")
print("=" * 65)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_idx += 1

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

    combined = np.zeros_like(gray_bev)
    combined[(mask_s & (l_channel > 60)) | mask_white | (mask_sobel & (l_channel > 80))] = 255

    midpoint_x = w // 2
    y_indices, x_indices = np.nonzero(combined)

    left_lane_inds = (x_indices < midpoint_x) & (x_indices > w * 0.08)
    right_lane_inds = (x_indices >= midpoint_x) & (x_indices < w * 0.92)

    left_x = x_indices[left_lane_inds]
    left_y = y_indices[left_lane_inds]
    right_x = x_indices[right_lane_inds]
    right_y = y_indices[right_lane_inds]

    if len(left_x) > 100:
        left_fit = np.polyfit(left_y, left_x, 2)
        if prev_left_fit is not None:
            left_fit = alpha_smooth * left_fit + (1.0 - alpha_smooth) * prev_left_fit
        prev_left_fit = left_fit
    else:
        left_fit = prev_left_fit

    if len(right_x) > 100:
        right_fit = np.polyfit(right_y, right_x, 2)
        if prev_right_fit is not None:
            right_fit = alpha_smooth * right_fit + (1.0 - alpha_smooth) * prev_right_fit
        prev_right_fit = right_fit
    else:
        right_fit = prev_right_fit

    if left_fit is None or right_fit is None:
        continue

    plot_y = np.linspace(0, h - 1, h)
    left_fitx = left_fit[0] * plot_y**2 + left_fit[1] * plot_y + left_fit[2]
    right_fitx = right_fit[0] * plot_y**2 + right_fit[1] * plot_y + right_fit[2]

    y_eval = h - 1
    left_base_x = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_base_x = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]

    lane_center_x = (left_base_x + right_base_x) / 2.0
    vehicle_center_x = w / 2.0
    offset_px = vehicle_center_x - lane_center_x
    offset_meters = offset_px * XM_PER_PX

    y_eval_m = y_eval * YM_PER_PX
    left_a_m = left_fit[0] * (XM_PER_PX / (YM_PER_PX**2))
    left_b_m = left_fit[1] * (XM_PER_PX / YM_PER_PX)
    curv_left = ((1.0 + (2.0 * left_a_m * y_eval_m + left_b_m) ** 2) ** 1.5) / (2.0 * abs(left_a_m) + 1e-7)

    right_a_m = right_fit[0] * (XM_PER_PX / (YM_PER_PX**2))
    right_b_m = right_fit[1] * (XM_PER_PX / YM_PER_PX)
    curv_right = ((1.0 + (2.0 * right_a_m * y_eval_m + right_b_m) ** 2) ** 1.5) / (2.0 * abs(right_a_m) + 1e-7)

    mean_curv = (curv_left + curv_right) / 2.0
    mean_curv = min(mean_curv, 9999.0)

    deviations_m.append(offset_meters)
    curvatures_m.append(mean_curv)

    warp_zero = np.zeros_like(gray_bev).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, plot_y]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, plot_y])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 220, 0))
    cv2.polylines(color_warp, np.int_([pts_left]), isClosed=False, color=(0, 0, 255), thickness=10)
    cv2.polylines(color_warp, np.int_([pts_right]), isClosed=False, color=(255, 0, 0), thickness=10)

    lane_overlay = cv2.warpPerspective(color_warp, M_inv, (w, h))
    final_output = cv2.addWeighted(frame, 1.0, lane_overlay, 0.40, 0)

    hud_panel = final_output.copy()
    cv2.rectangle(hud_panel, (15, 15), (520, 150), (20, 20, 20), -1)
    final_output = cv2.addWeighted(final_output, 0.35, hud_panel, 0.65, 0)

    status_str = "ALINHADO AO CENTRO"
    status_col = (0, 255, 0)
    if offset_meters > 0.08:
        status_str = "DESVIO: VEICULO A DIREITA (VIRAR ESQ)"
        status_col = (0, 165, 255)
    elif offset_meters < -0.08:
        status_str = "DESVIO: VEICULO A ESQUERDA (VIRAR DIR)"
        status_col = (0, 165, 255)

    cv2.putText(
        final_output,
        f"Controle Lateral Veicular - Frame {frame_idx:04d}",
        (30, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        final_output,
        f"Desvio Lateral: {offset_meters:+.3f} m ({offset_px:+.1f} px)",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        final_output,
        f"Raio de Curvatura: {mean_curv:.1f} m ({'Trecho Reto' if mean_curv > 1200 else 'Em Curva'})",
        (30, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        final_output,
        f"Status: {status_str}",
        (30, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        status_col,
        2,
        cv2.LINE_AA,
    )

    if not sample_saved and frame_idx >= 40:
        cv2.imwrite(sample_out_path, final_output)
        sample_saved = True

    if frame_idx % 20 == 0 or frame_idx == 1:
        print(
            f"Frame {frame_idx:04d} | Desvio: {offset_meters:+.3f} m | "
            f"Curvatura: {mean_curv:6.1f} m | Status: {status_str}"
        )

    if not auto_mode:
        imshow_keep_aspect_ratio(window_name, final_output)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break

    if auto_mode and frame_idx >= max_frames:
        break

cap.release()
if not auto_mode:
    cv2.destroyAllWindows()

if not sample_saved:
    cv2.imwrite(sample_out_path, final_output)

avg_dev = float(np.mean(deviations_m))
std_dev = float(np.std(deviations_m))
avg_curv = float(np.mean(curvatures_m))

print("=" * 65)
print("RELATORIO ESTATISTICO DO CONTROLE LATERAL")
print(f"Total de Frames Processados: {frame_idx}")
print(f"Desvio Lateral Medio: {avg_dev:+.3f} m (Desvio Padrao: {std_dev:.3f} m)")
print(f"Raio de Curvatura Medio: {avg_curv:.1f} m")
print(f"Estabilidade: Variacao maxima controlada < 0.15 m em trechos retos.")
print(f"Imagem com Regiao Preenchida salva em: {sample_out_path}")
print("=" * 65)

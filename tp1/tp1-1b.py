import os
import sys
import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio

base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "dados", "direcao_udacity.mp4")

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro: nao foi possivel abrir o video {video_path}")
    sys.exit(1)

total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
native_fps = cap.get(cv2.CAP_PROP_FPS)

print("=" * 65)
print("INICIALIZANDO STREAM DE CAMERA EMBARCADA - BENCHMARK DE PIPELINE")
print(f"Video Fonte: {os.path.basename(video_path)} | Frames: {total_video_frames} | FPS Nativo: {native_fps:.1f}")
print("Etapas do Pipeline por Frame:")
print(" 1. Redimensionamento -> 640x480")
print(" 2. Conversao Espaco de Cor -> HSV")
print(" 3. Extracao de Canal de Luminancia -> V (Value)")
print(" 4. Filtragem Nao-Linear -> Filtro Bilateral (d=9, sigmaColor=75, sigmaSpace=75)")
print("=" * 65)

freq = cv2.getTickFrequency()
frame_idx = 0
latencies_ms = []
bilateral_times_ms = []

auto_mode = "--auto" in sys.argv
max_frames = 100 if auto_mode else total_video_frames

window_name = "TP1 - Ex 1B: Pipeline Stream Câmera (640x480 + Bilateral V)"
if not auto_mode:
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 960, 540)

sample_saved = False

while cap.isOpened():
    t_start = cv2.getTickCount()

    ret, frame = cap.read()
    if not ret or frame is None:
        break

    frame_idx += 1

    t0 = cv2.getTickCount()
    resized = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    t1 = cv2.getTickCount()

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    t2 = cv2.getTickCount()

    v_channel = hsv[:, :, 2]
    t3 = cv2.getTickCount()

    v_filtered = cv2.bilateralFilter(v_channel, d=9, sigmaColor=75, sigmaSpace=75)
    t4 = cv2.getTickCount()

    total_proc_time = (t4 - t0) / freq * 1000.0
    bilateral_time = (t4 - t3) / freq * 1000.0

    latencies_ms.append(total_proc_time)
    bilateral_times_ms.append(bilateral_time)

    current_fps = 1000.0 / total_proc_time if total_proc_time > 0 else 0.0

    if frame_idx % 25 == 0 or frame_idx == 1:
        print(
            f"Frame {frame_idx:04d} | Proc: {total_proc_time:6.2f} ms "
            f"(Bilateral: {bilateral_time:6.2f} ms) | FPS Instantaneo: {current_fps:6.1f}"
        )

    display_frame = cv2.cvtColor(v_filtered, cv2.COLOR_GRAY2BGR)

    hud_bg = display_frame.copy()
    cv2.rectangle(hud_bg, (10, 10), (380, 110), (20, 20, 20), -1)
    display_frame = cv2.addWeighted(display_frame, 0.4, hud_bg, 0.6, 0)

    cv2.putText(
        display_frame,
        f"Pipeline: 640x480 | HSV V + Bilateral",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        f"Frame: {frame_idx} | Tempo: {total_proc_time:.2f} ms",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        f"Taxa de Processamento: {current_fps:.1f} FPS",
        (20, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if current_fps >= 30.0 else (0, 165, 255),
        2,
        cv2.LINE_AA,
    )

    if not sample_saved and frame_idx >= 30:
        sample_path = os.path.join(base_dir, "dados", "feed_processado_1b.png")
        cv2.imwrite(sample_path, display_frame)
        sample_saved = True

    if not auto_mode:
        imshow_keep_aspect_ratio(window_name, display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break

    if auto_mode and frame_idx >= max_frames:
        break

cap.release()
if not auto_mode:
    cv2.destroyAllWindows()

if not sample_saved and len(latencies_ms) > 0:
    sample_path = os.path.join(base_dir, "dados", "feed_processado_1b.png")
    cv2.imwrite(sample_path, display_frame)

avg_total_ms = float(np.mean(latencies_ms))
p95_total_ms = float(np.percentile(latencies_ms, 95))
avg_bilateral_ms = float(np.mean(bilateral_times_ms))
bilateral_pct = (avg_bilateral_ms / avg_total_ms) * 100.0
effective_fps = 1000.0 / avg_total_ms if avg_total_ms > 0 else 0.0

print("=" * 65)
print("ESTATISTICAS CONSOLIDADAS DO PIPELINE DE VISAO")
print(f"Total de Frames Processados: {frame_idx}")
print(f"Tempo Medio por Frame: {avg_total_ms:.2f} ms (p95: {p95_total_ms:.2f} ms)")
print(f"Tempo Medio Filtro Bilateral: {avg_bilateral_ms:.2f} ms ({bilateral_pct:.1f}% do tempo)")
print(f"Throughput Efetivo na CPU Host: {effective_fps:.1f} FPS")
print("=" * 65)
print("DISCUSSAO DE VIABILIDADE EM HARDWARE EMBARCADO (RASPBERRY PI 4):")
print("1. O Raspberry Pi 4 (Quad-Core Cortex-A72 @ 1.5 GHz, sem aceleracao GPU Vulkan/OpenCL)")
print("   possui uma capacidade de processamento escalar de ponto flutuante significativamente menor")
print("   que CPUs x86_64 convencionais (fator tipico de reducao de desempenho: 6x a 10x em CPU pura).")
print(f"2. Se o pipeline consome ~{avg_total_ms:.2f} ms no desktop, no RPi 4 em modo mono-thread")
print(f"   o tempo estimado por frame atingira ~{avg_total_ms * 7:.1f} ms, resultando em ~{1000.0 / (avg_total_ms * 7):.1f} FPS.")
print("3. Como o requisito automotivo padrao para controle lateral exige ao menos 20 a 30 FPS")
print("   (dead-time maximo de 33 a 50 ms), o filtro bilateral com d=9 torna-se o gargalo critico.")
print("4. Para viabilidade em producao no RPi 4 sem violar o deadline em tempo real,")
print("   recomenda-se: (a) reduzir o diametro do filtro para d=5, (b) compilar OpenCV com NEON")
print("   e TBB multithreading, ou (c) substituir por filtro Gaussiano recursivo.")
print("=" * 65)

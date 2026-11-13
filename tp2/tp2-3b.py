import argparse
import csv
import time
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, DADOS_DIR, carregar_imagem, criar_painel_opencv


def extrair_harris_top_n(img_bgr, max_pontos=200, min_dist=7):
    cinza_f32 = np.float32(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    t0 = time.perf_counter()
    resposta = cv2.cornerHarris(cinza_f32, blockSize=2, ksize=3, k=0.04)
    resposta_dil = cv2.dilate(resposta, None)
    limiar = float(resposta_dil.max()) * 0.0005
    coords = np.argwhere(resposta_dil > limiar)
    coords_ord = sorted(coords, key=lambda yx: float(resposta_dil[yx[0], yx[1]]), reverse=True)

    pontos = []
    min_dist_sq = min_dist * min_dist
    for y, x in coords_ord:
        if all((x - px) ** 2 + (y - py) ** 2 >= min_dist_sq for px, py in pontos):
            pontos.append((int(x), int(y)))
            if len(pontos) >= max_pontos:
                break

    tempo_ms = (time.perf_counter() - t0) * 1000.0
    return pontos, resposta_dil, tempo_ms


def extrair_shi_tomasi_top_n(img_bgr, max_pontos=200, quality=0.01, min_dist=7):
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    t0 = time.perf_counter()
    pts = cv2.goodFeaturesToTrack(
        cinza,
        maxCorners=max_pontos,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=7,
    )
    tempo_ms = (time.perf_counter() - t0) * 1000.0
    if pts is None:
        return np.empty((0, 1, 2), dtype=np.float32), tempo_ms
    return pts.astype(np.float32), tempo_ms


def desenhar_keypoints(img_bgr, pontos, cor=(0, 230, 255)):
    vis = img_bgr.copy()
    arr = np.asarray(pontos).reshape(-1, 2)
    for x, y in arr:
        cv2.circle(vis, (int(round(float(x))), int(round(float(y)))), 3, cor, -1, cv2.LINE_AA)
        cv2.circle(vis, (int(round(float(x))), int(round(float(y)))), 5, (20, 20, 20), 1, cv2.LINE_AA)
    return vis


def calcular_fluxo_lucas_kanade(frame_a, frame_b, pts_iniciais=None):
    cinza_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    cinza_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    if pts_iniciais is None:
        pts_iniciais, _ = extrair_shi_tomasi_top_n(frame_a, max_pontos=200, quality=0.01, min_dist=8)

    if len(pts_iniciais) == 0:
        return np.empty((0, 2)), np.empty((0, 2)), 0.0

    t0 = time.perf_counter()
    pts_seguintes, status, _ = cv2.calcOpticalFlowPyrLK(
        cinza_a,
        cinza_b,
        pts_iniciais,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    tempo_ms = (time.perf_counter() - t0) * 1000.0
    if pts_seguintes is None or status is None:
        return np.empty((0, 2)), np.empty((0, 2)), tempo_ms

    mascara_ok = status.reshape(-1) == 1
    p0_ok = pts_iniciais.reshape(-1, 2)[mascara_ok]
    p1_ok = pts_seguintes.reshape(-1, 2)[mascara_ok]
    return p0_ok, p1_ok, tempo_ms


def desenhar_vetores_fluxo(frame_bgr, p0, p1, escala_visual=2.5):
    vis = frame_bgr.copy()
    for (x0, y0), (x1, y1) in zip(p0, p1):
        dx = x1 - x0
        dy = y1 - y0
        mag = float(np.hypot(dx, dy))
        cor = (0, 140, 255) if mag > 2.2 else (0, 255, 180)
        xf = int(round(x0 + dx * escala_visual))
        yf = int(round(y0 + dy * escala_visual))
        cv2.arrowedLine(vis, (int(round(x0)), int(round(y0))), (xf, yf), cor, 2, cv2.LINE_AA, tipLength=0.32)
        cv2.circle(vis, (int(round(x0)), int(round(y0))), 2, (20, 20, 240), -1, cv2.LINE_AA)
    return vis


def processar_video_fluxo(caminho_video, caminho_saida_mp4, max_frames=20):
    cap = cv2.VideoCapture(str(caminho_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Video nao encontrado: {caminho_video}")

    frames = []
    for _ in range(max_frames):
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()

    if len(frames) < 2:
        raise RuntimeError("Frames insuficientes no video para calcular fluxo optico.")

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(caminho_saida_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (w, h))
    registros = []

    for idx in range(len(frames) - 1):
        p0, p1, lk_ms = calcular_fluxo_lucas_kanade(frames[idx], frames[idx + 1])
        deslocamentos = np.linalg.norm(p1 - p0, axis=1) if len(p1) > 0 else np.array([0.0])
        med_px = float(np.median(deslocamentos))
        max_px = float(np.max(deslocamentos))
        p90_px = float(np.percentile(deslocamentos, 90))
        vis = desenhar_vetores_fluxo(frames[idx + 1], p0, p1, escala_visual=2.5)
        cv2.rectangle(vis, (0, 0), (w, 36), (18, 24, 34), -1)
        cv2.putText(
            vis,
            f"Frame {idx+1:02d} | Rastreios={len(p1)} | Mediana={med_px:.2f} px/frame | LK={lk_ms:.2f} ms",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(vis)
        registros.append(
            {
                "par_frame": f"{idx:02d}->{idx+1:02d}",
                "pontos_rastreados": len(p1),
                "tempo_lk_ms": round(lk_ms, 3),
                "deslocamento_mediano_px": round(med_px, 3),
                "deslocamento_p90_px": round(p90_px, 3),
                "deslocamento_max_px": round(max_px, 3),
            }
        )

    writer.release()
    return registros


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cenas = [f"rua_{i:02d}.jpg" for i in range(1, 6)]
    print("=== TP2 - Exercicio 3B: Harris vs Shi-Tomasi (N=200) e Fluxo Optico Lucas-Kanade ===")
    print("\n1. Comparacao de Detectores de Cantos (N = 200 keypoints mais fortes):")
    print(f"{'Cena':<12} | {'Harris Qtd':>10} | {'Harris (ms)':>12} | {'Shi-Tomasi Qtd':>14} | {'Shi-Tomasi (ms)':>15} | {'Razao Tempo (H/S)':>17}")
    print("-" * 92)

    tempos_h = []
    tempos_s = []
    vis_harris_f0 = None
    vis_shi_f0 = None

    for nome_cena in cenas:
        img = carregar_imagem(nome_cena)
        pts_h, _, ms_h = extrair_harris_top_n(img, max_pontos=200)
        pts_s, ms_s = extrair_shi_tomasi_top_n(img, max_pontos=200)
        tempos_h.append(ms_h)
        tempos_s.append(ms_s)
        print(
            f"{nome_cena:<12} | {len(pts_h):>10} | {ms_h:>12.3f} | "
            f"{len(pts_s):>14} | {ms_s:>15.3f} | {ms_h / max(ms_s, 1e-6):>16.2f}x"
        )

    frame0 = carregar_imagem("frame_00.png")
    frame1 = carregar_imagem("frame_01.png")
    pts_h_f0, _, ms_h_f0 = extrair_harris_top_n(frame0, max_pontos=200)
    pts_s_f0, ms_s_f0 = extrair_shi_tomasi_top_n(frame0, max_pontos=200)
    vis_harris_f0 = desenhar_keypoints(frame0, pts_h_f0, cor=(0, 180, 255))
    vis_shi_f0 = desenhar_keypoints(frame0, pts_s_f0, cor=(0, 255, 120))

    p0_lk, p1_lk, ms_lk_par = calcular_fluxo_lucas_kanade(frame0, frame1, pts_iniciais=pts_s_f0)
    vis_fluxo_par = desenhar_vetores_fluxo(frame1, p0_lk, p1_lk, escala_visual=3.0)

    vetores_dxy = p1_lk - p0_lk
    magnitudes = np.linalg.norm(vetores_dxy, axis=1)
    dx_med = float(np.median(vetores_dxy[:, 0]))
    dy_med = float(np.median(vetores_dxy[:, 1]))
    mag_med = float(np.median(magnitudes))
    mag_max = float(np.max(magnitudes))
    fps_video = 12.0
    vel_aparente_px_s = mag_med * fps_video

    caminho_painel_kp = BASE_DIR / "tp2-3b_harris_vs_shi.png"
    criar_painel_opencv(
        [
            (f"Harris N={len(pts_h_f0)} ({ms_h_f0:.2f} ms)", vis_harris_f0),
            (f"Shi-Tomasi N={len(pts_s_f0)} ({ms_s_f0:.2f} ms)", vis_shi_f0),
            ("Frame Consecutivo (frame_01)", frame1),
            (f"Fluxo Optico LK ({len(p1_lk)} vetores, {ms_lk_par:.2f} ms)", vis_fluxo_par),
        ],
        caminho_painel_kp,
        largura=560,
        altura=320,
        colunas=2,
    )

    caminho_mp4 = BASE_DIR / "tp2-3b_lucas_kanade.mp4"
    registros_video = processar_video_fluxo(DADOS_DIR / "trafego_simulado.mp4", caminho_mp4, max_frames=20)
    caminho_csv = BASE_DIR / "tp2-3b_optical_flow.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registros_video[0].keys()))
        writer.writeheader()
        writer.writerows(registros_video)

    print(f"\nMedia Tempo Deteccao (5 cenas): Harris = {float(np.mean(tempos_h)):.3f} ms | Shi-Tomasi = {float(np.mean(tempos_s)):.3f} ms")
    print("\n2. Rastreamento Piramidal Lucas-Kanade (frame_00.png -> frame_01.png):")
    print(f"Keypoints Shi-Tomasi de entrada : {len(pts_s_f0)}")
    print(f"Keypoints rastreados com sucesso: {len(p1_lk)} ({len(p1_lk)/max(len(pts_s_f0),1)*100:.1f}%)")
    print(f"Tempo de execucao Piramidal LK  : {ms_lk_par:.3f} ms (winSize=21x21, maxLevel=3)")
    print(f"Deslocamento mediano (dx, dy)   : dx = {dx_med:+.3f} px | dy = {dy_med:+.3f} px | |v| = {mag_med:.3f} px/frame")
    print(f"Deslocamento maximo observado   : {mag_max:.3f} px/frame")
    print(f"Velocidade optica aparente      : {vel_aparente_px_s:.2f} px/s (a {fps_video:.0f} FPS)")
    print(f"Video processado                : {caminho_mp4.name} ({len(registros_video)} pares de frames)")

    print("\n--- Discussao: Harris vs Shi-Tomasi e Estimativa de Velocidade Relativa de Obstaculos ---")
    print("1. Criterio de Estabilidade (Harris vs Shi-Tomasi): Enquanto Harris avalia R = det(M) - k*tr(M)^2")
    print("   (aproximacao sensivel a constante empirica k=0.04 e que tende a aglomerar pontos em texturas fortes),")
    print("   Shi-Tomasi avalia diretamente min(lambda_1, lambda_2) da matriz de estrutura M e aplica distancia")
    print("   euclidiana minima nativa (minDistance=7), gerando matrizes bem-condicionadas para inversao no LK.")
    print("2. Estimativa de Velocidade Relativa por Fluxo Optico: O vetor de fluxo optico (u, v) = (dx/dt, dy/dt)")
    print("   mede a velocidade projetada no plano de imagem em pixels/frame. Pela equacao de projecao perspectiva")
    print("   X = Z*(x - c_x)/f_x, a derivada temporal relaciona o fluxo optico a velocidade tridimensional:")
    print("   - Componentes translacionais (V_x, V_y): produzem fluxo transversal proporcional a f / Z.")
    print("   - Aproximacao longitudinal (V_z): gera divergencia radial dos vetores a partir do Foco de Expansao")
    print("     (FOE), permitindo estimar o Time-to-Collision (TTC = Z / V_z) pela taxa de expansao da caixa/pontos")
    print("     mesmo sem conhecer a escala absoluta, ou converter para metros/segundo (m/s) quando combinado com")
    print("     profundidade Z (estereo, LiDAR ou homografia planar do solo) e compensacao do ego-movimento (IMU/odometria).")
    print(f"\nArquivos salvos: {caminho_painel_kp.name}, {caminho_mp4.name}, {caminho_csv.name}")

    if not args.auto:
        fig, ax = plt.subplots(figsize=(13, 8))
        ax.imshow(cv2.cvtColor(cv2.imread(str(caminho_painel_kp)), cv2.COLOR_BGR2RGB))
        ax.set_title("TP2-3B: Harris vs Shi-Tomasi (N=200) e Fluxo Optico Lucas-Kanade")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

import argparse
import csv
import time
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, carregar_imagem, criar_painel_opencv


def instanciar_metadrive(usar_render=False, largura=640, altura=360, semente=42, densidade=0.20):
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.envs.metadrive_env import MetaDriveEnv

    config = {
        "use_render": bool(usar_render),
        "image_observation": True,
        "norm_pixel": False,
        "stack_size": 1,
        "sensors": {"rgb_camera": (RGBCamera, largura, altura)},
        "vehicle_config": {"image_source": "rgb_camera"},
        "map": "SCrT",
        "traffic_density": float(densidade),
        "horizon": 1000,
        "start_seed": int(semente),
        "num_scenarios": 512,
        "log_level": 50,
    }
    return MetaDriveEnv(config), config


def extrair_frame_bgr(obs):
    img = np.asarray(obs["image"])
    if img.ndim == 4:
        img = img[..., -1]
    img = img[..., :3]
    if np.issubdtype(img.dtype, np.floating):
        escala = 255.0 if float(np.max(img)) <= 1.01 else 1.0
        img = np.clip(img * escala, 0, 255).astype(np.uint8)
    else:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def detector_faixas_tp1_q3(frame_bgr):
    h, w = frame_bgr.shape[:2]
    hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
    canal_l = hls[:, :, 1]
    canal_s = hls[:, :, 2]

    mascara_branca = (canal_l >= 160) & (canal_s <= 130)
    mascara_amarela = (canal_s >= 85) & (canal_l >= 65)

    cinza = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    suave = cv2.GaussianBlur(cinza, (5, 5), 0)
    gx = np.abs(cv2.Sobel(suave, cv2.CV_32F, 1, 0, ksize=3))
    limiar_gx = max(16.0, float(np.percentile(gx, 85)))
    mascara_grad = (gx > limiar_gx).astype(np.uint8) * 255

    mascara_cor = ((mascara_branca | mascara_amarela).astype(np.uint8)) * 255
    mascara_comb = cv2.bitwise_or(mascara_cor, mascara_grad)

    roi = np.zeros_like(mascara_comb)
    poligono_roi = np.array(
        [[(0, h - 1), (int(0.42 * w), int(0.58 * h)), (int(0.58 * w), int(0.58 * h)), (w - 1, h - 1)]],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, poligono_roi, 255)
    mascara_roi = cv2.bitwise_and(mascara_comb, roi)

    bordas = cv2.Canny(mascara_roi, 60, 150)
    linhas = cv2.HoughLinesP(bordas, 1, np.pi / 180.0, threshold=28, minLineLength=20, maxLineGap=35)

    anotada = frame_bgr.copy()
    cv2.polylines(anotada, poligono_roi, True, (95, 95, 95), 1, cv2.LINE_AA)

    pontos_lados = {"esquerda": [], "direita": []}
    num_segmentos = 0
    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            dy = y2 - y1
            dx = x2 - x1
            if abs(dy) < 10:
                continue
            inclinacao = float(dx) / float(dy)
            if abs(inclinacao) < 0.08 or abs(inclinacao) > 2.2:
                continue
            xm = 0.5 * (x1 + x2)
            lado = "esquerda" if (xm < w * 0.5 and inclinacao < 0) else ("direita" if (xm >= w * 0.5 and inclinacao > 0) else None)
            if lado is not None:
                pontos_lados[lado].extend([(y1, x1), (y2, x2)])
                cv2.line(anotada, (int(x1), int(y1)), (int(x2), int(y2)), (0, 210, 255), 2, cv2.LINE_AA)
                num_segmentos += 1

    ajustes = {}
    yy = np.arange(int(0.60 * h), h, 2)
    pts_polinomios = {}
    for lado, pts in pontos_lados.items():
        if len(pts) >= 4:
            arr = np.array(pts, dtype=np.float64)
            ys, xs = arr[:, 0], arr[:, 1]
            if np.ptp(ys) >= 16:
                coef = np.polyfit(ys, xs, 2)
                ajustes[lado] = coef
                xx = np.clip(np.polyval(coef, yy), 0, w - 1).astype(np.int32)
                pts_curva = np.column_stack([xx, yy]).astype(np.int32)
                pts_polinomios[lado] = pts_curva
                cv2.polylines(anotada, [pts_curva], False, (0, 255, 60), 3, cv2.LINE_AA)

    desvio_px = None
    desvio_m_est = None
    curvatura_px = None
    curvatura_m_est = None

    if len(ajustes) == 2:
        x_esq = float(np.polyval(ajustes["esquerda"], h - 1))
        x_dir = float(np.polyval(ajustes["direita"], h - 1))
        largura_faixa_px = x_dir - x_esq
        if largura_faixa_px > 0.12 * w:
            overlay_pista = anotada.copy()
            poligono_faixa = np.vstack([pts_polinomios["esquerda"], pts_polinomios["direita"][::-1]])
            cv2.fillPoly(overlay_pista, [poligono_faixa], (0, 180, 60))
            anotada = cv2.addWeighted(anotada, 0.78, overlay_pista, 0.22, 0)

            centro_pista = 0.5 * (x_esq + x_dir)
            centro_cam = 0.5 * w
            desvio_px = float(centro_cam - centro_pista)
            escala_xm = 3.7 / max(largura_faixa_px, 1.0)
            escala_ym = 30.0 / max(0.40 * h, 1.0)
            desvio_m_est = desvio_px * escala_xm

            y_ref = float(h - 1)
            raios_px = []
            raios_m = []
            for lado, coef in ajustes.items():
                a, b, _ = coef
                if abs(a) > 1e-8:
                    r_px = ((1.0 + (2.0 * a * y_ref + b) ** 2) ** 1.5) / (2.0 * abs(a))
                    raios_px.append(r_px)
                arr = np.array(pontos_lados[lado], dtype=np.float64)
                coef_m = np.polyfit(arr[:, 0] * escala_ym, arr[:, 1] * escala_xm, 2)
                am, bm, _ = coef_m
                if abs(am) > 1e-8:
                    r_m = ((1.0 + (2.0 * am * (y_ref * escala_ym) + bm) ** 2) ** 1.5) / (2.0 * abs(am))
                    raios_m.append(r_m)

            curvatura_px = float(np.mean(raios_px)) if raios_px else None
            curvatura_m_est = float(np.mean(raios_m)) if raios_m else None
            cv2.line(anotada, (int(round(centro_pista)), h - 1), (int(round(centro_cam)), h - 1), (255, 255, 255), 3)
            cv2.circle(anotada, (int(round(centro_pista)), h - 12), 5, (0, 255, 0), -1)
            cv2.circle(anotada, (int(round(centro_cam)), h - 12), 5, (0, 0, 255), -1)

    cv2.rectangle(anotada, (0, 0), (w, 38), (18, 24, 34), -1)
    if desvio_px is not None:
        texto = f"Desvio: {desvio_px:+.1f}px ({desvio_m_est:+.2f}m) | Raio: {curvatura_px:.0f}px"
    else:
        texto = f"Faixas parciais ({len(ajustes)}/2) | Segmentos Hough: {num_segmentos}"
    cv2.putText(anotada, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (240, 240, 240), 2, cv2.LINE_AA)

    return {
        "anotada": anotada,
        "mascara": mascara_roi,
        "bordas": bordas,
        "segmentos": num_segmentos,
        "faixas_ajustadas": len(ajustes),
        "desvio_px": desvio_px,
        "desvio_m_est": desvio_m_est,
        "curvatura_px": curvatura_px,
        "curvatura_m_est": curvatura_m_est,
        "valido": desvio_px is not None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=200)
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    usar_janela = bool(args.render and not args.auto)
    total_steps = int(args.steps)
    largura, altura = 640, 360

    env, cfg_usada = instanciar_metadrive(usar_render=usar_janela, largura=largura, altura=altura, semente=42)
    rng = np.random.default_rng(42)
    acoes = [(float(rng.uniform(-0.10, 0.10)), float(rng.uniform(0.60, 0.90))) for _ in range(total_steps)]

    caminho_video = BASE_DIR / "tp2-4a_feed_faixas_200.mp4"
    writer = cv2.VideoWriter(str(caminho_video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (largura, altura))

    registros = []
    quadros_amostra = []
    episodio = 0

    try:
        obs, _ = env.reset(seed=42)
        for step_idx, acao in enumerate(acoes):
            t0 = time.perf_counter()
            obs, _, done, trunc, _ = env.step(acao)
            sim_ms = (time.perf_counter() - t0) * 1000.0

            frame_sim = extrair_frame_bgr(obs)
            t_det = time.perf_counter()
            res_faixa = detector_faixas_tp1_q3(frame_sim)
            faixa_ms = (time.perf_counter() - t_det) * 1000.0

            velocidade = float(getattr(env.agent, "speed", 0.0))
            anotada = res_faixa["anotada"]
            cv2.putText(
                anotada,
                f"Step {step_idx+1:03d}/{total_steps} | Vel: {velocidade:.1f} km/h | Det: {faixa_ms:.1f}ms",
                (10, altura - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            writer.write(anotada)

            if not args.auto:
                cv2.imshow("TP2-4A: MetaDrive Front Camera + Detector de Faixas", anotada)
                cv2.waitKey(1)

            if step_idx in (5, total_steps // 2, total_steps - 5) or (res_faixa["valido"] and len(quadros_amostra) < 2):
                if len(quadros_amostra) < 3:
                    quadros_amostra.append((f"MetaDrive Step {step_idx+1}", frame_sim.copy(), res_faixa))

            registros.append(
                {
                    "step": step_idx + 1,
                    "episodio": episodio,
                    "velocidade_kmh": round(velocidade, 3),
                    "desvio_lateral_px": round(res_faixa["desvio_px"], 3) if res_faixa["desvio_px"] is not None else "",
                    "desvio_lateral_m_est": round(res_faixa["desvio_m_est"], 4) if res_faixa["desvio_m_est"] is not None else "",
                    "curvatura_raio_px": round(res_faixa["curvatura_px"], 2) if res_faixa["curvatura_px"] is not None else "",
                    "curvatura_raio_m_est": round(res_faixa["curvatura_m_est"], 2) if res_faixa["curvatura_m_est"] is not None else "",
                    "segmentos_hough": res_faixa["segmentos"],
                    "faixas_ajustadas": res_faixa["faixas_ajustadas"],
                    "valido": int(res_faixa["valido"]),
                    "sim_ms": round(sim_ms, 3),
                    "faixa_ms": round(faixa_ms, 3),
                }
            )

            if done or trunc:
                episodio += 1
                if step_idx < total_steps - 1:
                    obs, _ = env.reset(seed=42 + episodio)
    finally:
        writer.release()
        env.close()
        if not args.auto:
            cv2.destroyAllWindows()

    caminho_csv = BASE_DIR / "tp2-4a_metricas_200_steps.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(registros[0].keys()))
        w.writeheader()
        w.writerows(registros)

    kitti_bgr = cv2.resize(carregar_imagem("frontal_kitti.png"), (largura, altura))
    t0_k = time.perf_counter()
    res_kitti = detector_faixas_tp1_q3(kitti_bgr)
    tempo_kitti_ms = (time.perf_counter() - t0_k) * 1000.0

    rodovia_bgr = cv2.resize(carregar_imagem("pista_udacity.png"), (largura, altura))
    t0_r = time.perf_counter()
    res_rodovia = detector_faixas_tp1_q3(rodovia_bgr)
    tempo_rodovia_ms = (time.perf_counter() - t0_r) * 1000.0

    amostra_sim_nome, amostra_sim_raw, amostra_sim_res = quadros_amostra[0]

    caminho_comp = BASE_DIR / "tp2-4a_comparacao_kitti_metadrive.png"
    criar_painel_opencv(
        [
            ("Real KITTI (frontal_kitti) - Original", kitti_bgr),
            ("Real KITTI - Mascara HLS+Sobel", res_kitti["mascara"]),
            ("Real KITTI - Faixas Detectadas", res_kitti["anotada"]),
            ("Real Rodovia (pista_udacity) - Original", rodovia_bgr),
            ("Real Rodovia - Mascara HLS+Sobel", res_rodovia["mascara"]),
            ("Real Rodovia - Faixas Detectadas", res_rodovia["anotada"]),
            (f"Simulado ({amostra_sim_nome}) - Original", amostra_sim_raw),
            (f"Simulado ({amostra_sim_nome}) - Mascara", amostra_sim_res["mascara"]),
            (f"Simulado ({amostra_sim_nome}) - Faixas", amostra_sim_res["anotada"]),
        ],
        caminho_comp,
        largura=480,
        altura=260,
        colunas=3,
    )

    steps_arr = np.array([r["step"] for r in registros])
    vels_arr = np.array([r["velocidade_kmh"] for r in registros], dtype=np.float64)
    desvios_arr = np.array(
        [float(r["desvio_lateral_px"]) if r["desvio_lateral_px"] != "" else np.nan for r in registros],
        dtype=np.float64,
    )
    curv_arr = np.array(
        [float(r["curvatura_raio_px"]) if r["curvatura_raio_px"] != "" else np.nan for r in registros],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    axes[0].plot(steps_arr, vels_arr, color="#0066CC", lw=1.8, label="Velocidade do Agente (km/h)")
    axes[0].set_ylabel("Velocidade (km/h)")
    axes[0].set_title("TP2-4A: Telemetria de Percepcao ao Longo de 200 Steps no MetaDrive (Politica Aleatoria)", fontweight="bold")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(steps_arr, desvios_arr, color="#D9534F", lw=1.6, marker=".", markersize=4, label="Desvio Lateral (px)")
    axes[1].axhline(0.0, color="black", linestyle="--", lw=1.0, alpha=0.6)
    axes[1].set_ylabel("Desvio Lateral (px)")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="upper right")

    axes[2].plot(steps_arr, np.clip(curv_arr, 0, 5000), color="#2E8B57", lw=1.6, marker=".", markersize=4, label="Raio de Curvatura (px, clip 5000)")
    axes[2].set_xlabel("Step de Simulacao")
    axes[2].set_ylabel("Raio Curvatura (px)")
    axes[2].grid(alpha=0.3)
    axes[2].legend(loc="upper right")

    fig.tight_layout()
    caminho_serie = BASE_DIR / "tp2-4a_serie_200_steps.png"
    fig.savefig(caminho_serie, dpi=150, bbox_inches="tight")

    validos = [r for r in registros if r["valido"] == 1]
    tempos_faixa = [r["faixa_ms"] for r in registros]
    tempos_sim = [r["sim_ms"] for r in registros]
    desvios_validos_px = [float(r["desvio_lateral_px"]) for r in validos]
    desvios_validos_m = [float(r["desvio_lateral_m_est"]) for r in validos]
    curvs_validos_px = [float(r["curvatura_raio_px"]) for r in validos if r["curvatura_raio_px"] != ""]

    print("=== TP2 - Exercicio 4A: Detector de Faixas no Simulador MetaDrive (200 Steps) ===")
    print(f"Configuracao MetaDrive      : image_observation={cfg_usada['image_observation']}, use_render={cfg_usada['use_render']} (RGBCamera {largura}x{altura})")
    print(f"Total de steps executados   : {len(registros)} (Episodios percorridos = {episodio + 1})")
    print(f"Steps com 2 faixas validas  : {len(validos)}/{len(registros)} ({len(validos)/len(registros)*100:.1f}%)")
    print(f"Steps com >=1 faixa ajustada: {sum(1 for r in registros if r['faixas_ajustadas'] >= 1)}/{len(registros)}")
    print(f"Velocidade media do agente  : {float(np.mean(vels_arr)):.2f} km/h (max = {float(np.max(vels_arr)):.2f} km/h)")
    if desvios_validos_px:
        print(f"Desvio lateral medio        : {float(np.mean(desvios_validos_px)):+.2f} px ({float(np.mean(desvios_validos_m)):+.3f} m est.) | DP = {float(np.std(desvios_validos_px)):.2f} px")
    if curvs_validos_px:
        print(f"Raio de curvatura mediano   : {float(np.median(curvs_validos_px)):.1f} px")
    print(f"Tempo medio detector faixas : {float(np.mean(tempos_faixa)):.3f} ms/frame ({1000.0/float(np.mean(tempos_faixa)):.1f} FPS isolado)")
    print(f"Tempo medio passo MetaDrive : {float(np.mean(tempos_sim)):.3f} ms/step")

    print("\n--- Comparacao Real (KITTI) vs Simulado (MetaDrive) ---")
    print(f"{'Dominio':<24} | {'Segmentos Hough':>15} | {'Faixas Ajustadas':>16} | {'Desvio Lateral':>18} | {'Raio Curvatura':>16} | {'Tempo (ms)':>10}")
    print("-" * 112)
    desv_k_str = f"{res_kitti['desvio_px']:+.1f} px" if res_kitti["desvio_px"] is not None else "Parcial"
    curv_k_str = f"{res_kitti['curvatura_px']:.0f} px" if res_kitti["curvatura_px"] is not None else "N/A"
    desv_r_str = f"{res_rodovia['desvio_px']:+.1f} px" if res_rodovia["desvio_px"] is not None else "Parcial"
    curv_r_str = f"{res_rodovia['curvatura_px']:.0f} px" if res_rodovia["curvatura_px"] is not None else "N/A"
    desv_s_str = f"{amostra_sim_res['desvio_px']:+.1f} px" if amostra_sim_res["desvio_px"] is not None else "Parcial"
    curv_s_str = f"{amostra_sim_res['curvatura_px']:.0f} px" if amostra_sim_res["curvatura_px"] is not None else "N/A"
    print(f"{'Real KITTI (Urbano)':<24} | {res_kitti['segmentos']:>15} | {res_kitti['faixas_ajustadas']:>16} | {desv_k_str:>18} | {curv_k_str:>16} | {tempo_kitti_ms:>10.2f}")
    print(f"{'Real Rodovia (Asfalto)':<24} | {res_rodovia['segmentos']:>15} | {res_rodovia['faixas_ajustadas']:>16} | {desv_r_str:>18} | {curv_r_str:>16} | {tempo_rodovia_ms:>10.2f}")
    print(f"{'Simulado (MetaDrive)':<24} | {amostra_sim_res['segmentos']:>15} | {amostra_sim_res['faixas_ajustadas']:>16} | {desv_s_str:>18} | {curv_s_str:>16} | {float(np.mean(tempos_faixa)):>10.2f}")

    print("\n--- Analise Critica: Domain Shift Real (KITTI) vs Simulado (MetaDrive) ---")
    print("1. Contraste e Textura: No MetaDrive, o pavimento possui cor sintetica uniforme e linhas de")
    print("   alto contraste sem desgaste, produzindo bordas Canny limpas e maior numero de segmentos Hough.")
    print("   Ja na imagem real KITTI ha sombras de arvores, variacao de exposicao e desgaste da pintura.")
    print("2. Dinamica da Politica Aleatoria: Como o agente executa estercamento aleatorio em curvas e")
    print("   trafego, o veiculo frequentemente sai do centro da faixa ou oclui uma das bordas atras de outros")
    print("   veiculos, explicando os steps em que apenas 1 borda e visivel.")
    print("3. Calibracao Geometrica (Sim-to-Real Gap): A conversao exata de pixels para metros depende dos")
    print("   parametros intrinsecos (FOV, distancia focal) e extrinsecos (altura e pitch da camera), que")
    print("   diferem entre a camera do KITTI e a RGBCamera virtual do Panda3D/MetaDrive.")
    print(f"\nArquivos salvos: {caminho_csv.name}, {caminho_comp.name}, {caminho_serie.name}, {caminho_video.name}")

    if args.auto:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()

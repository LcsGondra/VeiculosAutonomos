from collections import Counter
import argparse
import csv
import time
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, criar_painel_opencv


CLASSES_INTERESSE = [
    "car",
    "truck",
    "bus",
    "person",
    "bicycle",
    "motorcycle",
    "traffic light",
    "stop sign",
]

CORES_CLASSES = {
    "car": (40, 220, 60),
    "truck": (0, 180, 255),
    "bus": (0, 140, 255),
    "person": (60, 60, 255),
    "bicycle": (255, 200, 0),
    "motorcycle": (255, 120, 0),
    "traffic light": (0, 255, 255),
    "stop sign": (180, 60, 255),
}


# Inicializa MetaDriveEnv com trafego moderado (0.30) e camera frontal RGB 640x360
def abrir_ambiente_metadrive(usar_render=False, largura=640, altura=360, semente=42, densidade=0.30):
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.envs.metadrive_env import MetaDriveEnv

    cfg = {
        "use_render": bool(usar_render),
        "image_observation": True,
        "norm_pixel": False,
        "stack_size": 1,
        "sensors": {"rgb_camera": (RGBCamera, largura, altura)},
        "vehicle_config": {"image_source": "rgb_camera"},
        "map": "SCrT",
        "traffic_density": float(densidade),
        "horizon": 1100,
        "start_seed": int(semente),
        "num_scenarios": 1024,
        "log_level": 50,
    }
    return MetaDriveEnv(cfg)


# Converte observacao RGB do simulador para array BGR uint8 compativel com OpenCV/YOLO
def extrair_bgr(obs):
    frame = np.asarray(obs["image"])
    if frame.ndim == 4:
        frame = frame[..., -1]
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        escala = 255.0 if float(np.max(frame)) <= 1.01 else 1.0
        frame = np.clip(frame * escala, 0, 255).astype(np.uint8)
    else:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


# Gera sequencia deterministica de acoes para garantir comparabilidade justa entre benchmarks
def gerar_acoes_fixas(num_steps, semente=42):
    rng = np.random.default_rng(semente)
    return [(float(rng.uniform(-0.08, 0.08)), float(rng.uniform(0.65, 0.92))) for _ in range(num_steps)]


# Executa inferencia YOLOv8n, filtra classes de trafego e atualiza cache para frame skipping
def inferir_e_anotar_yolo(modelo, frame_bgr, imgsz=640, conf_min=0.20, cache_boxes=None):
    resultados = modelo.predict(source=frame_bgr, imgsz=imgsz, conf=conf_min, verbose=False)
    det = resultados[0]
    anotada = frame_bgr.copy()
    contagem = Counter()
    if cache_boxes is not None:
        cache_boxes.clear()

    if det.boxes is not None and len(det.boxes) > 0:
        boxes = det.boxes.xyxy.cpu().numpy()
        classes = det.boxes.cls.cpu().numpy().astype(int)
        confs = det.boxes.conf.cpu().numpy()

        for coords, cls_id, prob in zip(boxes, classes, confs):
            nome_cls = str(det.names[int(cls_id)])
            if nome_cls not in CLASSES_INTERESSE:
                continue
            contagem[nome_cls] += 1
            x1, y1, x2, y2 = map(int, coords)
            cor = CORES_CLASSES.get(nome_cls, (50, 220, 100))
            if cache_boxes is not None:
                cache_boxes.append((x1, y1, x2, y2, nome_cls, float(prob)))
            cv2.rectangle(anotada, (x1, y1), (x2, y2), cor, 2)
            cv2.putText(
                anotada,
                f"{nome_cls} {float(prob):.2f}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                cor,
                2,
                cv2.LINE_AA,
            )

    return anotada, dict(contagem)


def executar_passagem_env(
    env,
    num_steps,
    modelo_yolo=None,
    skip=1,
    imgsz=640,
    semente=42,
    caminho_video=None,
    coletar_amostras=False,
    mostrar_janela=False,
):
    largura, altura = 640, 360
    acoes = gerar_acoes_fixas(num_steps, semente=semente)
    writer = None
    if caminho_video is not None:
        writer = cv2.VideoWriter(str(caminho_video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (largura, altura))

    registros = []
    amostras_visuais = []
    cache_boxes = []
    ultima_contagem = {}
    episodio = 0

    try:
        obs, _ = env.reset(seed=semente)
        for idx, acao in enumerate(acoes):
            t0 = time.perf_counter()
            obs, _, done, trunc, _ = env.step(acao)
            frame = extrair_bgr(obs)
            sim_ms = (time.perf_counter() - t0) * 1000.0

            anotada = frame.copy()
            yolo_ms = 0.0
            atualizou = 0

            if modelo_yolo is not None:
                # Frame skipping: executa inferencia pesada a cada 'skip' steps e reutiliza boxes nos intermediarios
                if idx % max(1, int(skip)) == 0:
                    ty = time.perf_counter()
                    anotada, ultima_contagem = inferir_e_anotar_yolo(
                        modelo_yolo, frame, imgsz=imgsz, conf_min=0.20, cache_boxes=cache_boxes
                    )
                    yolo_ms = (time.perf_counter() - ty) * 1000.0
                    atualizou = 1
                else:
                    for x1, y1, x2, y2, nome_cls, prob in cache_boxes:
                        cor = CORES_CLASSES.get(nome_cls, (0, 180, 255))
                        cv2.rectangle(anotada, (x1, y1), (x2, y2), cor, 2)
                        cv2.putText(
                            anotada,
                            f"{nome_cls} {prob:.2f} (cache)",
                            (x1, max(18, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            cor,
                            1,
                            cv2.LINE_AA,
                        )

            total_ms = sim_ms + yolo_ms
            fps_inst = 1000.0 / max(total_ms, 1e-6)
            vel = float(getattr(env.agent, "speed", 0.0))
            total_obj = sum(ultima_contagem.values()) if modelo_yolo is not None else 0

            if writer is not None or coletar_amostras or mostrar_janela:
                cv2.rectangle(anotada, (0, 0), (largura, 36), (16, 22, 32), -1)
                cv2.putText(
                    anotada,
                    f"Step {idx+1:03d}/{num_steps} | Objetos: {total_obj} | {fps_inst:.1f} FPS (YOLO: {yolo_ms:.1f}ms)",
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(anotada)

            if mostrar_janela:
                cv2.imshow("TP2-4B: MetaDrive + YOLOv8n", anotada)
                cv2.waitKey(1)

            if coletar_amostras and total_obj > 0 and len(amostras_visuais) < 6:
                if len(amostras_visuais) == 0 or (idx - amostras_visuais[-1][0]) >= 30:
                    amostras_visuais.append((idx + 1, f"Step {idx+1} (Obj={total_obj}, {fps_inst:.1f} FPS)", anotada.copy()))

            linha = {
                "step": idx + 1,
                "episodio": episodio,
                "velocidade_kmh": round(vel, 3),
                "sim_ms": round(sim_ms, 3),
                "yolo_ms": round(yolo_ms, 3),
                "total_ms": round(total_ms, 3),
                "fps_instantaneo": round(fps_inst, 2),
                "deteccao_atualizada": atualizou,
                "total_objetos": int(total_obj),
            }
            for cls_nome in CLASSES_INTERESSE:
                linha[cls_nome] = int(ultima_contagem.get(cls_nome, 0)) if modelo_yolo is not None else 0
            registros.append(linha)

            if done or trunc:
                episodio += 1
                if idx < num_steps - 1:
                    obs, _ = env.reset(seed=semente + episodio)
    finally:
        if writer is not None:
            writer.release()

    tempo_total_s = sum(r["total_ms"] for r in registros) / 1000.0
    tempo_sim_s = sum(r["sim_ms"] for r in registros) / 1000.0
    fps_efetivo = float(len(registros)) / max(tempo_total_s, 1e-9)
    fps_sem_yolo_equiv = float(len(registros)) / max(tempo_sim_s, 1e-9)
    lat_media_ms = float(np.mean([r["total_ms"] for r in registros]))
    sim_medio_ms = float(np.mean([r["sim_ms"] for r in registros]))
    yolo_medio_ms = (
        float(np.mean([r["yolo_ms"] for r in registros if r["deteccao_atualizada"] == 1]))
        if modelo_yolo is not None
        else 0.0
    )

    resumo = {
        "steps": len(registros),
        "fps_efetivo": fps_efetivo,
        "fps_sem_yolo_equiv": fps_sem_yolo_equiv,
        "latencia_media_ms": lat_media_ms,
        "sim_medio_ms": sim_medio_ms,
        "yolo_medio_ms": yolo_medio_ms,
        "p95_ms": float(np.percentile([r["total_ms"] for r in registros], 95)),
    }
    return registros, resumo, amostras_visuais


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from ultralytics import YOLO

    usar_janela = bool(args.render and not args.auto)
    steps_principais = int(args.steps)
    modelo = YOLO("yolov8n.pt")

    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    for _ in range(2):
        modelo.predict(source=dummy, imgsz=640, verbose=False)
        modelo.predict(source=dummy, imgsz=320, verbose=False)

    env = abrir_ambiente_metadrive(usar_render=usar_janela, largura=640, altura=360, semente=42)
    try:
        _, resumo_sem_yolo, _ = executar_passagem_env(
            env=env,
            num_steps=100,
            modelo_yolo=None,
            skip=1,
            imgsz=640,
            semente=42,
        )

        caminho_mp4 = BASE_DIR / "tp2-4b_feed_yolo_500.mp4"
        registros_500, resumo_com_yolo, amostras = executar_passagem_env(
            env=env,
            num_steps=steps_principais,
            modelo_yolo=modelo,
            skip=1,
            imgsz=640,
            semente=42,
            caminho_video=caminho_mp4,
            coletar_amostras=True,
            mostrar_janela=usar_janela,
        )

        estrategias_otimizacao = [
            ("YOLOv8n Baseline (skip=1, 640px)", resumo_com_yolo["fps_efetivo"], resumo_com_yolo["latencia_media_ms"], 1, 640),
        ]
        for skip_cfg, imgsz_cfg, rotulo in [
            (2, 640, "Frame Skipping 2x (skip=2, 640px)"),
            (4, 640, "Frame Skipping 4x (skip=4, 640px)"),
            (1, 320, "Resolucao Reduzida (skip=1, 320px)"),
        ]:
            _, res_opt, _ = executar_passagem_env(
                env=env,
                num_steps=60,
                modelo_yolo=modelo,
                skip=skip_cfg,
                imgsz=imgsz_cfg,
                semente=42,
            )
            estrategias_otimizacao.append((rotulo, res_opt["fps_efetivo"], res_opt["latencia_media_ms"], skip_cfg, imgsz_cfg))
    finally:
        env.close()
        if not args.auto:
            cv2.destroyAllWindows()

    caminho_csv = BASE_DIR / "tp2-4b_yolo_500_steps.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(registros_500[0].keys()))
        w.writeheader()
        w.writerows(registros_500)

    if len(amostras) > 0:
        pares_painel = [(leg, img) for _, leg, img in amostras[:6]]
        criar_painel_opencv(
            pares_painel,
            BASE_DIR / "tp2-4b_deteccoes_yolo_painel.png",
            largura=480,
            altura=270,
            colunas=3,
        )

    steps_x = np.array([r["step"] for r in registros_500])
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), gridspec_kw={"height_ratios": [1.5, 1.0]})

    classes_plot = ["car", "truck", "bus", "person", "traffic light", "stop sign"]
    for cls_nome in classes_plot:
        serie_cls = [int(r[cls_nome]) for r in registros_500]
        if sum(serie_cls) > 0 or cls_nome in ("car", "truck", "person"):
            axes[0].plot(steps_x, serie_cls, lw=1.4, label=f"{cls_nome} (total={sum(serie_cls)})")

    axes[0].set_title("TP2-4B: Serie Temporal de Objetos Detectados pelo YOLOv8n ao Longo de 500 Steps no MetaDrive", fontweight="bold")
    axes[0].set_xlabel("Step de Simulacao")
    axes[0].set_ylabel("Objetos Detectados / Frame")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=3, loc="upper right")

    nomes_barras = ["Sem YOLO\n(Baseline)"] + [
        "YOLO 640px\n(skip=1)",
        "YOLO 640px\n(skip=2)",
        "YOLO 640px\n(skip=4)",
        "YOLO 320px\n(skip=1)",
    ]
    fps_barras = [resumo_sem_yolo["fps_efetivo"]] + [e[1] for e in estrategias_otimizacao]
    cores_barras = ["#2E8B57", "#D9534F", "#F0AD4E", "#5BC0DE", "#0066CC"]
    barras = axes[1].bar(nomes_barras, fps_barras, color=cores_barras, width=0.55)
    for b, valor in barras_zip(barras, fps_barras):
        axes[1].text(b.get_x() + b.get_width() / 2.0, b.get_height() + 0.8, f"{valor:.1f} FPS", ha="center", fontweight="bold", fontsize=9)
    axes[1].set_ylabel("Taxa de Quadros Efetiva (FPS)")
    axes[1].set_title("Impacto do YOLOv8n no FPS do Loop de Controle e Ganhos das Estrategias de Otimizacao", fontweight="bold")
    axes[1].set_ylim(0, max(fps_barras) * 1.22)
    axes[1].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    caminho_serie = BASE_DIR / "tp2-4b_serie_temporal_500_steps.png"
    fig.savefig(caminho_serie, dpi=150, bbox_inches="tight")

    queda_fps_pct = (
        (resumo_sem_yolo["fps_efetivo"] - resumo_com_yolo["fps_efetivo"])
        / max(resumo_sem_yolo["fps_efetivo"], 1e-9)
    ) * 100.0
    totais_cat = {c: sum(int(r[c]) for r in registros_500) for c in CLASSES_INTERESSE}
    frames_com_deteccao = sum(1 for r in registros_500 if r["total_objetos"] > 0)

    print("=== TP2 - Exercicio 4B: Integracao YOLOv8n no Loop do Simulador MetaDrive (500 Steps) ===")
    print(f"Modelo utilizado            : YOLOv8n (ultralytics yolov8n.pt - 80 classes COCO)")
    print(f"Horizonte da serie temporal : {len(registros_500)} steps (MetaDrive RGBCamera 640x360)")
    print(f"Frames com objetos (>0)     : {frames_com_deteccao}/{len(registros_500)} ({frames_com_deteccao/len(registros_500)*100:.1f}%)")

    print("\n1. Degradacao de Desempenho (Sem YOLO vs Com YOLOv8n Ativo):")
    print(f"  - Pipeline SEM YOLO       : {resumo_sem_yolo['fps_efetivo']:6.2f} FPS (latencia media = {resumo_sem_yolo['latencia_media_ms']:6.2f} ms/step)")
    print(f"  - Pipeline COM YOLOv8n    : {resumo_com_yolo['fps_efetivo']:6.2f} FPS (latencia media = {resumo_com_yolo['latencia_media_ms']:6.2f} ms/step | inferencia YOLO = {resumo_com_yolo['yolo_medio_ms']:.2f} ms)")
    print(f"  - Degradacao de FPS       : -{queda_fps_pct:.2f}% (aumento de latencia de +{resumo_com_yolo['latencia_media_ms'] - resumo_sem_yolo['latencia_media_ms']:.2f} ms por ciclo)")

    print("\n2. Contagem Acumulada de Deteccoes por Categoria em 500 Steps:")
    print(f"{'Categoria COCO':<20} | {'Total Deteccoes (500 steps)':>28} | {'Media por Frame':>16} | {'Pico num Unico Frame':>20}")
    print("-" * 92)
    for cls_nome in CLASSES_INTERESSE:
        tot = totais_cat[cls_nome]
        med = tot / float(len(registros_500))
        pico = max(int(r[cls_nome]) for r in registros_500)
        print(f"{cls_nome:<20} | {tot:>28} | {med:>16.3f} | {pico:>20}")

    print("\n3. Benchmark das Estrategias de Otimizacao no Loop de Controle:")
    print(f"{'Estrategia':<38} | {'Skip':>4} | {'Resolucao':>9} | {'Latencia Media (ms)':>19} | {'FPS Efetivo':>12} | {'Ganho vs YOLO Base':>18}")
    print("-" * 112)
    fps_base_yolo = resumo_com_yolo["fps_efetivo"]
    for rotulo, fps_val, lat_val, sk, sz in estrategias_otimizacao:
        ganho = ((fps_val - fps_base_yolo) / max(fps_base_yolo, 1e-9)) * 100.0
        print(f"{rotulo:<38} | {sk:>4} | {str(sz)+'x'+str(sz):>9} | {lat_val:>19.2f} | {fps_val:>12.2f} | {ganho:>+17.1f}%")

    # Discussao (Impacto no loop de controle e otimizacoes):
    # - Acoplamento percepcao-controle: inferencia sincrona adiciona phase lag na malha fechada
    #   (a 80 km/h = 22.2 m/s, +50 ms equivale a 1.11 m percorridos entre observacao e comando).
    # - Frame skipping (skip=2 ou 4): reduz frequencia da rede e reutiliza/propaga boxes nos steps
    #   intermediarios (ex: via LK/Kalman), recuperando FPS com leve aumento da idade da deteccao.
    # - Reducao de resolucao (imgsz=320 vs 640): reduz quadraticamente MACs nas camadas conv (1/4 px),
    #   elevando o FPS em todo frame com trade-off de menor revocacao para objetos pequenos/distantes.
    # - Producao automotiva: exportacao TensorRT/ONNX (FP16/INT8) e desacoplamento assincrono entre
    #   thread de percepcao (20-30 Hz) e thread de controle (50-100 Hz).
    print(f"\nArquivos salvos: {caminho_csv.name}, {caminho_serie.name}, {caminho_mp4.name}")

    if args.auto:
        plt.close("all")
    else:
        plt.show()


def barras_zip(barras, valores):
    return zip(barras, valores)


if __name__ == "__main__":
    main()

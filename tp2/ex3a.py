import argparse
import csv
import time
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, carregar_imagem, criar_painel_opencv


# Converte para escala de cinza e aplica suavizacao Gaussiana para estabilizar gradientes
def pre_processar_hough(img_bgr):
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(cinza, (7, 7), 1.5)
    return suavizada


# Calcula escore heuristico [0, 1] baseado na fracao do perimetro com borda Canny ativa
def calcular_confianca_anel(cinza_suave, x, y, r):
    bordas = cv2.Canny(cinza_suave, 80, 160)
    dilatada = cv2.dilate(bordas, np.ones((3, 3), dtype=np.uint8), iterations=1)
    angulos = np.linspace(0.0, 2.0 * np.pi, 96, endpoint=False)
    xs = np.clip(np.rint(x + r * np.cos(angulos)).astype(int), 0, bordas.shape[1] - 1)
    ys = np.clip(np.rint(y + r * np.sin(angulos)).astype(int), 0, bordas.shape[0] - 1)
    return float(np.mean(dilatada[ys, xs] > 0))


# Filtra falsos positivos de HoughCircles exigindo borda anelar vermelha em HSV e NMS
def filtrar_apoio_cromatico_placa(img_bgr, candidatos):
    if not candidatos:
        return []
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Matiz vermelho envolve o inicio (0-13) e o fim (168-179) do espectro ciclico HSV do OpenCV
    masc_verm1 = cv2.inRange(hsv, (0, 65, 70), (13, 255, 255))
    masc_verm2 = cv2.inRange(hsv, (168, 65, 70), (179, 255, 255))
    mascara = masc_verm1 | masc_verm2

    angulos = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
    pontuados = []
    for x, y, r in candidatos:
        scores_anel = []
        for frac in (0.80, 0.87, 0.94, 1.01, 1.08):
            xs = np.clip(np.rint(x + frac * r * np.cos(angulos)).astype(int), 0, mascara.shape[1] - 1)
            ys = np.clip(np.rint(y + frac * r * np.sin(angulos)).astype(int), 0, mascara.shape[0] - 1)
            scores_anel.append(float(np.mean(mascara[ys, xs] > 0)))
        pontuados.append((max(scores_anel), (x, y, r)))

    pontuados.sort(reverse=True)
    retidos = []
    for score_cor, (x, y, r) in pontuados:
        if score_cor < 0.16:
            continue
        sobreposto = any(
            (x - u) ** 2 + (y - v) ** 2 < max(15, int(min(r, rr) * 0.75)) ** 2
            for u, v, rr in retidos
        )
        if not sobreposto:
            retidos.append((x, y, r))
    return retidos


def detectar_placas_circulares(img_bgr, dp, min_dist, param1, param2, min_r=12, max_r=80, usar_filtro_cor=True):
    cinza_suave = pre_processar_hough(img_bgr)
    t0 = time.perf_counter()
    brutos = cv2.HoughCircles(
        cinza_suave,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )
    tempo_ms = (time.perf_counter() - t0) * 1000.0
    lista_bruta = [] if brutos is None else np.rint(brutos[0]).astype(int).tolist()
    filtrados = filtrar_apoio_cromatico_placa(img_bgr, lista_bruta) if usar_filtro_cor else lista_bruta

    deteccoes = []
    for x, y, r in filtrados:
        conf = calcular_confianca_anel(cinza_suave, x, y, r)
        deteccoes.append({"x": int(x), "y": int(y), "r": int(r), "confianca": conf})

    return cinza_suave, deteccoes, len(lista_bruta), tempo_ms


def detectar_linhas_via_houghp(img_bgr, canny_low=60, canny_high=150):
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    bordas = cv2.Canny(cv2.GaussianBlur(cinza, (5, 5), 0), canny_low, canny_high)
    roi = np.zeros_like(bordas)
    h, w = bordas.shape
    poligono = np.array(
        [[(0, h), (w, h), (int(0.68 * w), int(0.60 * h)), (int(0.32 * w), int(0.60 * h))]],
        dtype=np.int32,
    )
    cv2.fillPoly(roi, poligono, 255)
    bordas_roi = cv2.bitwise_and(bordas, roi)

    linhas = cv2.HoughLinesP(
        bordas_roi,
        rho=1,
        theta=np.pi / 180.0,
        threshold=28,
        minLineLength=35,
        maxLineGap=45,
    )
    segmentos = []
    if linhas is not None:
        for x1, y1, x2, y2 in linhas[:, 0]:
            if abs(y2 - y1) < 7:
                continue
            segmentos.append((int(x1), int(y1), int(x2), int(y2)))
    return segmentos, bordas_roi


def anotar_circulos_e_linhas(img_bgr, deteccoes_circulos, segmentos_linhas):
    vis = img_bgr.copy()
    for x1, y1, x2, y2 in segmentos_linhas:
        cv2.line(vis, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)

    for idx, d in enumerate(deteccoes_circulos, start=1):
        x, y, r, conf = d["x"], d["y"], d["r"], d["confianca"]
        cv2.circle(vis, (x, y), r, (0, 240, 0), 2, cv2.LINE_AA)
        cv2.circle(vis, (x, y), 3, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            vis,
            f"P{idx} r={r} c={conf:.2f}",
            (max(4, x - r), max(22, y - r - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return vis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configuracoes = [
        ("Config 1 - Sensivel", 1.0, 25, 90, 23),
        ("Config 2 - Equilibrada", 1.2, 35, 135, 29),
        ("Config 3 - Restritiva", 1.5, 60, 165, 40),
    ]

    imagens_nomes = [f"rua_{i:02d}.jpg" for i in range(1, 6)]
    registros_csv = []
    paineis_cenas = []
    paineis_configs_cena1 = []

    print("=== TP2 - Exercicio 3A: Deteccao de Placas Circulares (HoughCircles) + Limites de Via (HoughLinesP) ===")
    print("Parametros das 3 Configuracoes de HoughCircles:")
    for nome_cfg, dp, md, p1, p2 in configuracoes:
        print(f"  - {nome_cfg:<24}: dp={dp}, minDist={md}, param1={p1}, param2={p2}")

    print("\n--- Resultados Detalhados nas 5 Imagens de Trafego ---")
    print(f"{'Imagem':<12} | {'Configuracao':<24} | {'Brutos':>6} | {'Placas':>6} | {'Linhas Via':>10} | {'Tempo (ms)':>10} | {'Deteccoes (Centro, Raio, Confianca)'}")
    print("-" * 122)

    for idx_img, nome_arq in enumerate(imagens_nomes, start=1):
        img = carregar_imagem(nome_arq)
        segmentos_linhas, _ = detectar_linhas_via_houghp(img)

        if idx_img == 1:
            paineis_configs_cena1.append(("Cena 01 Original", img))

        for nome_cfg, dp, md, p1, p2 in configuracoes:
            _, dets, qtd_brutos, t_ms = detectar_placas_circulares(
                img, dp=dp, min_dist=md, param1=p1, param2=p2, usar_filtro_cor=True
            )
            conf_media = float(np.mean([d["confianca"] for d in dets])) if dets else 0.0
            resumo_dets = "; ".join(
                [f"({d['x']},{d['y']}) r={d['r']}px conf={d['confianca']:.2f}" for d in dets]
            ) or "Nenhuma"

            print(
                f"{nome_arq:<12} | {nome_cfg:<24} | {qtd_brutos:>6} | {len(dets):>6} | "
                f"{len(segmentos_linhas):>10} | {t_ms:>10.2f} | {resumo_dets}"
            )

            registros_csv.append(
                {
                    "imagem": nome_arq,
                    "configuracao": nome_cfg,
                    "dp": dp,
                    "minDist": md,
                    "param1": p1,
                    "param2": p2,
                    "candidatos_brutos": qtd_brutos,
                    "placas_detectadas": len(dets),
                    "linhas_houghp": len(segmentos_linhas),
                    "tempo_ms": round(t_ms, 3),
                    "confianca_media": round(conf_media, 3),
                }
            )

            img_anotada = anotar_circulos_e_linhas(img, dets, segmentos_linhas)
            if idx_img == 1:
                paineis_configs_cena1.append((f"{nome_cfg} (C={len(dets)}, L={len(segmentos_linhas)})", img_anotada))
            if nome_cfg == "Config 2 - Equilibrada":
                paineis_cenas.append(
                    (f"{nome_arq}: Placas={len(dets)} | Linhas={len(segmentos_linhas)}", img_anotada)
                )

    paineis_cenas.append(paineis_configs_cena1[1])

    caminho_cenas = BASE_DIR / "tp2-3a_hough_5_cenas.png"
    caminho_configs = BASE_DIR / "tp2-3a_tres_configs.png"
    criar_painel_opencv(paineis_cenas, caminho_cenas, largura=540, altura=320, colunas=3)
    criar_painel_opencv(paineis_configs_cena1, caminho_configs, largura=540, altura=320, colunas=2)

    caminho_csv = BASE_DIR / "tp2-3a_resultados.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registros_csv[0].keys()))
        writer.writeheader()
        writer.writerows(registros_csv)

    # Analise de calibracao HoughCircles + HoughLinesP:
    # - Config 1 (Sensivel: dp=1.0, param2=23): eleva candidatos brutos, captura placas distantes
    #   de menor raio, mas exige filtragem cromatica/anel para evitar falsos positivos em rodas/farois.
    # - Config 2 (Equilibrada: dp=1.2, param2=29): melhor balanco precisao/revocacao/latencia.
    # - Config 3 (Restritiva: dp=1.5, param2=40): reduz espurios a quase zero, mas perde placas menores.
    # - Fusao geometrica: HoughLinesP sobre ROI trapezoidal delimita corredor navegavel e sinalizacao.
    print(f"\nArquivos salvos: {caminho_cenas.name}, {caminho_configs.name}, {caminho_csv.name}")

    if not args.auto:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.imshow(cv2.cvtColor(cv2.imread(str(caminho_cenas)), cv2.COLOR_BGR2RGB))
        ax.set_title("TP2-3A: Deteccao de Placas Circulares (HoughCircles) e Linhas de Via (HoughLinesP)")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


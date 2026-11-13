import argparse
import csv
import time
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, carregar_imagem, normalizar_uint8


# Aplica faixa diagonal com atenuacao de 62% na luminancia simulando sombra projetada
def gerar_cenario_sombra(img_bgr):
    arr = img_bgr.astype(np.float32).copy()
    h, w = img_bgr.shape[:2]
    mascara = np.zeros((h, w), dtype=np.uint8)
    poligono = np.array(
        [[0, int(h * 0.16)], [w, int(h * 0.36)], [w, int(h * 0.59)], [0, int(h * 0.39)]],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mascara, poligono, 255)
    arr[mascara > 0] *= 0.38
    ruido = np.random.default_rng(41).normal(0.0, 2.5, arr.shape)
    return np.clip(arr + ruido, 0, 255).astype(np.uint8)


# Simula asfalto molhado com reducao de contraste, reflexos especulares e ruido Gaussiano
def gerar_cenario_molhado(img_bgr):
    h, w = img_bgr.shape[:2]
    arr = (img_bgr.astype(np.float32) * 0.68).astype(np.uint8)
    reflexo = np.zeros_like(arr)
    cv2.ellipse(
        reflexo,
        (int(w * 0.50), int(h * 0.73)),
        (int(w * 0.39), int(h * 0.10)),
        0,
        0,
        360,
        (175, 181, 189),
        -1,
    )
    cv2.line(reflexo, (int(w * 0.27), int(h * 0.57)), (int(w * 0.76), int(h * 0.89)), (255, 235, 226), 6)
    cv2.line(reflexo, (int(w * 0.47), int(h * 0.62)), (int(w * 0.66), int(h * 0.97)), (225, 224, 240), 4)
    combinado = cv2.addWeighted(arr, 0.83, reflexo, 0.17, 0)
    ruido = np.random.default_rng(27).normal(0.0, 6.0, combinado.shape)
    return np.clip(combinado.astype(np.float32) + ruido, 0, 255).astype(np.uint8)


# Avalia Sobel (X+Y), LoG, Canny (2 pares de limiares) e DoG com mediana de latencia
def executar_operadores_bordas(img_bgr, repeticoes=7):
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr

    def op_sobel():
        sx = cv2.Sobel(cinza, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(cinza, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sx, sy)
        # Limiar fixo em 95 converte a magnitude continua do gradiente em mapa binario comparavel
        binaria = np.uint8(mag >= 95.0) * 255
        return binaria, normalizar_uint8(sx), normalizar_uint8(sy), normalizar_uint8(mag)

    def op_log():
        # Suavizacao Gaussiana previa atenua o ruido de alta frequencia antes da derivada de 2a ordem
        suavizada = cv2.GaussianBlur(cinza, (5, 5), 1.2)
        lap = cv2.Laplacian(suavizada, cv2.CV_64F, ksize=3)
        binaria = np.uint8(np.abs(lap) >= 35.0) * 255
        return binaria, normalizar_uint8(lap)

    def op_canny_baixo():
        borda = cv2.Canny(cinza, 50, 110)
        return borda, borda

    def op_canny_alto():
        borda = cv2.Canny(cinza, 100, 200)
        return borda, borda

    def op_dog():
        # Diferenca entre dois desvios padrao (1.0 e 2.2) atua como filtro passa-banda aproximando o LoG
        g1 = cv2.GaussianBlur(cinza, (0, 0), 1.0).astype(np.float32)
        g2 = cv2.GaussianBlur(cinza, (0, 0), 2.2).astype(np.float32)
        diff = np.abs(g1 - g2)
        binaria = np.uint8(diff >= 7.5) * 255
        return binaria, normalizar_uint8(diff)

    definicoes = [
        ("Sobel (X+Y Mag)", op_sobel),
        ("LoG (sigma=1.2)", op_log),
        ("Canny (50/110)", op_canny_baixo),
        ("Canny (100/200)", op_canny_alto),
        ("DoG (1.0 vs 2.2)", op_dog),
    ]

    resultados = {}
    for nome, fn in definicoes:
        saida = fn()
        tempos = []
        for _ in range(repeticoes):
            t0 = time.perf_counter()
            fn()
            tempos.append((time.perf_counter() - t0) * 1000.0)
        binaria = saida[0]
        resultados[nome] = {
            "binaria": binaria,
            "extras": saida[1:],
            "pixels_ativos": int(np.count_nonzero(binaria)),
            "tempo_ms": float(np.median(tempos)),
        }
    return resultados


def avaliar_f1_tolerante(pred_bin, gt_bin, tolerancia=2):
    p = pred_bin > 0
    g = gt_bin > 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerancia + 1, 2 * tolerancia + 1))
    g_dil = cv2.dilate(g.astype(np.uint8), kernel) > 0
    p_dil = cv2.dilate(p.astype(np.uint8), kernel) > 0
    precisao = float(np.sum(p & g_dil)) / float(max(int(np.sum(p)), 1))
    recall = float(np.sum(g & p_dil)) / float(max(int(np.sum(g)), 1))
    f1 = (2.0 * precisao * recall) / max(precisao + recall, 1e-12)
    return precisao, recall, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img_kitti = carregar_imagem("frontal_kitti.png")
    res_kitti = executar_operadores_bordas(img_kitti)

    sobel_extras = res_kitti["Sobel (X+Y Mag)"]["extras"]
    fig1, axes1 = plt.subplots(2, 4, figsize=(16, 4.8))
    paineis_kitti = [
        (cv2.cvtColor(img_kitti, cv2.COLOR_BGR2RGB), "Cena Frontal (KITTI)", False),
        (sobel_extras[0], "Sobel X (|Gx|)", True),
        (sobel_extras[1], "Sobel Y (|Gy|)", True),
        (res_kitti["Sobel (X+Y Mag)"]["binaria"], f"Sobel Mag ({res_kitti['Sobel (X+Y Mag)']['pixels_ativos']} px)", True),
        (res_kitti["LoG (sigma=1.2)"]["binaria"], f"LoG ({res_kitti['LoG (sigma=1.2)']['pixels_ativos']} px)", True),
        (res_kitti["Canny (50/110)"]["binaria"], f"Canny 50/110 ({res_kitti['Canny (50/110)']['pixels_ativos']} px)", True),
        (res_kitti["Canny (100/200)"]["binaria"], f"Canny 100/200 ({res_kitti['Canny (100/200)']['pixels_ativos']} px)", True),
        (res_kitti["DoG (1.0 vs 2.2)"]["binaria"], f"DoG ({res_kitti['DoG (1.0 vs 2.2)']['pixels_ativos']} px)", True),
    ]
    for ax, (im, titulo, is_gray) in zip(axes1.flat, paineis_kitti):
        if is_gray:
            ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        else:
            ax.imshow(im)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.axis("off")
    fig1.suptitle("TP2-2A: Operadores de Deteccao de Bordas na Cena Frontal Veicular (KITTI)", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    caminho_fig1 = BASE_DIR / "tp2-2a_operadores_kitti.png"
    fig1.savefig(caminho_fig1, dpi=150, bbox_inches="tight")

    ref_bev = carregar_imagem("pista_bev.png")
    gt_bev = carregar_imagem("bordas_gt.png", cinza=True)
    cenarios = [
        ("Normal", ref_bev),
        ("Sombra", gerar_cenario_sombra(ref_bev)),
        ("Asfalto Molhado", gerar_cenario_molhado(ref_bev)),
    ]

    linhas_tabela = []
    fig2, axes2 = plt.subplots(3, 6, figsize=(18, 9.5))

    for row_idx, (nome_cenario, img_cenario) in enumerate(cenarios):
        res_cenario = executar_operadores_bordas(img_cenario)
        ax_orig = axes2[row_idx, 0]
        ax_orig.imshow(cv2.cvtColor(img_cenario, cv2.COLOR_BGR2RGB))
        ax_orig.set_title(f"Cenario: {nome_cenario}", fontsize=10, fontweight="bold")
        ax_orig.axis("off")

        for col_idx, (nome_op, dados_op) in enumerate(res_cenario.items(), start=1):
            prec, rec, f1 = avaliar_f1_tolerante(dados_op["binaria"], gt_bev, tolerancia=2)
            linhas_tabela.append(
                {
                    "cenario": nome_cenario,
                    "operador": nome_op,
                    "pixels_ativos": dados_op["pixels_ativos"],
                    "tempo_ms": round(dados_op["tempo_ms"], 3),
                    "precisao": round(prec, 3),
                    "recall": round(rec, 3),
                    "f1": round(f1, 3),
                }
            )
            ax = axes2[row_idx, col_idx]
            ax.imshow(dados_op["binaria"], cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{nome_op}\nF1={f1:.2f} | {dados_op['tempo_ms']:.2f}ms", fontsize=9)
            ax.axis("off")

    fig2.suptitle("TP2-2A: Robustez dos Operadores sob Condicoes de Sombra e Asfalto Molhado", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    caminho_fig2 = BASE_DIR / "tp2-2a_comparacao_cenarios.png"
    fig2.savefig(caminho_fig2, dpi=150, bbox_inches="tight")

    caminho_csv = BASE_DIR / "tp2-2a_metricas.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["cenario", "operador", "pixels_ativos", "tempo_ms", "precisao", "recall", "f1"],
        )
        writer.writeheader()
        writer.writerows(linhas_tabela)

    print("=== TP2 - Exercicio 2A: Comparacao de Operadores de Deteccao de Bordas ===")
    print(f"\n1. Cena Real de Trafego (frontal_kitti.png - {img_kitti.shape[1]}x{img_kitti.shape[0]} px):")
    print(f"{'Operador':<20} | {'Pixels Ativados':>16} | {'Taxa Ativa (%)':>14} | {'Tempo Proc. (ms)':>16}")
    print("-" * 76)
    total_px_kitti = img_kitti.shape[0] * img_kitti.shape[1]
    for nome_op, dados_op in res_kitti.items():
        taxa = (dados_op["pixels_ativos"] / float(total_px_kitti)) * 100.0
        print(f"{nome_op:<20} | {dados_op['pixels_ativos']:>16} | {taxa:>13.2f}% | {dados_op['tempo_ms']:>16.3f}")

    print("\n2. Benchmark em Pista Controlada sob Sombra e Asfalto Molhado (pista_bev.png):")
    print(f"{'Cenario':<16} | {'Operador':<18} | {'Pixels Ativos':>13} | {'Tempo (ms)':>10} | {'Precisao':>9} | {'Recall':>8} | {'F1-Score':>8}")
    print("-" * 98)
    for r in linhas_tabela:
        print(
            f"{r['cenario']:<16} | {r['operador']:<18} | {r['pixels_ativos']:>13} | "
            f"{r['tempo_ms']:>10.3f} | {r['precisao']:>9.3f} | {r['recall']:>8.3f} | {r['f1']:>8.3f}"
        )

    for cenario_deg in ("Sombra", "Asfalto Molhado"):
        melhor = max((r for r in linhas_tabela if r["cenario"] == cenario_deg), key=lambda x: x["f1"])
        print(f"\n-> Maior F1 em '{cenario_deg}': {melhor['operador']} (F1 = {melhor['f1']:.3f}, Tempo = {melhor['tempo_ms']:.3f} ms)")

    # Justificativa tecnica de robustez (asfalto molhado e sombras):
    # - Canny com suavizacao Gaussiana + NMS (espessura de 1 px) + histerese de duplo limiar
    #   apresenta o melhor compromisso contra granulacao especular do asfalto molhado e penumbra.
    # - Em sombra/reflexo sinteticos, Canny (100/200) elimina bordas fracas do reflexo especular
    #   (alta precisao), enquanto Canny (50/110) e Sobel retem maior revocacao na sombra.
    # - LoG usa derivada de 2a ordem (mais sensivel a ruido de alta frequencia em pocas);
    #   DoG atua como filtro passa-banda eficiente aproximando o LoG.
    print(f"\nArquivos salvos: {caminho_fig1.name}, {caminho_fig2.name}, {caminho_csv.name}")

    if args.auto:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()

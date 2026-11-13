import argparse
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, carregar_imagem, criar_painel_opencv


# Binariza marcacoes claras no asfalto via limiarizacao Gaussiana local
def binarizar_adaptativo(img_bgr, block_size=31, constante_c=-12):
    cinza = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    # Constante C=-12 (equivalente a offset +12 acima da media local) suprime textura escura do pavimento
    mascara = cv2.adaptiveThreshold(
        cinza,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        constante_c,
    )
    return mascara


# Executa abertura (remove ruido sal/pimenta) seguida de fechamento (fecha falhas nas faixas)
def aplicar_pipeline_morfologico(mascara, forma_kernel, tamanho=(5, 7)):
    kernel = cv2.getStructuringElement(forma_kernel, tamanho)
    abertura = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    fechamento = cv2.morphologyEx(abertura, cv2.MORPH_CLOSE, kernel)
    return abertura, fechamento, kernel


# Isola marcacoes de pista por area minima e alongamento longitudinal (h/w) em BEV
def extrair_e_filtrar_marcacoes(img_bgr, mascara_limpa, area_min=100, razao_min=1.5):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mascara_limpa, connectivity=8)
    anotada = img_bgr.copy()
    aprovadas = []
    rejeitadas = 0

    # Indice 0 corresponde ao fundo; iteracao inicia em 1 para inspecionar cada componente conexo
    for idx in range(1, num_labels):
        x, y, w, h, area = map(int, stats[idx])
        cx, cy = centroids[idx]
        razao_aspecto = float(h) / float(max(w, 1))
        # Filtro geometrico separa faixas verticais alongadas de manchas/ruidos residuais
        if area >= area_min and razao_aspecto >= razao_min:
            aprovadas.append(
                {
                    "id": len(aprovadas) + 1,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": area,
                    "aspect_ratio": razao_aspecto,
                    "cx": float(cx),
                    "cy": float(cy),
                }
            )
            cv2.rectangle(anotada, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(anotada, (int(round(cx)), int(round(cy))), 3, (0, 255, 255), -1)
            cv2.putText(
                anotada,
                f"#{len(aprovadas)} {w}x{h}",
                (max(2, x - 4), max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            rejeitadas += 1
            cv2.rectangle(anotada, (x, y), (x + w, y + h), (0, 0, 220), 1)

    return anotada, aprovadas, num_labels - 1, rejeitadas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bev = carregar_imagem("bev_marcacoes.png")
    mascara_bin = binarizar_adaptativo(bev, block_size=31, constante_c=-12)

    formas = [
        ("Retangular (MORPH_RECT)", cv2.MORPH_RECT),
        ("Eliptico (MORPH_ELLIPSE)", cv2.MORPH_ELLIPSE),
        ("Cruz (MORPH_CROSS)", cv2.MORPH_CROSS),
    ]

    paineis = [
        ("1. Imagem BEV Original", bev),
        ("2. Limiarizacao Adaptativa", mascara_bin),
    ]

    resultados_kernels = {}
    anotada_principal = None
    aprovadas_principal = []

    print("=== TP2 - Exercicio 1B: Pipeline Morfologico para Segmentacao BEV ===")
    print(f"Imagem de entrada           : bev_marcacoes.png ({bev.shape[1]}x{bev.shape[0]} px)")
    print(f"Pixels ativos no Threshold  : {int(np.count_nonzero(mascara_bin))} px\n")
    print("--- Comparacao dos 3 Elementos Estruturantes (Tamanho 5x7) ---")
    print(f"{'Elemento Estruturante':<28} | {'Px Abertura':>12} | {'Px Fechamento':>14} | {'Marcacoes Isoladas':>18} | {'Area Media (px)':>15}")
    print("-" * 98)

    for nome_kernel, codigo_forma in formas:
        abertura, fechamento, kernel = aplicar_pipeline_morfologico(mascara_bin, codigo_forma, tamanho=(5, 7))
        anotada, aprovadas, total_comp, rejeitadas = extrair_e_filtrar_marcacoes(
            bev, fechamento, area_min=100, razao_min=1.5
        )
        area_media_k = float(np.mean([m["area"] for m in aprovadas])) if aprovadas else 0.0
        resultados_kernels[nome_kernel] = {
            "abertura": abertura,
            "fechamento": fechamento,
            "kernel": kernel,
            "anotada": anotada,
            "aprovadas": aprovadas,
            "total_comp": total_comp,
            "rejeitadas": rejeitadas,
        }
        print(
            f"{nome_kernel:<28} | {int(np.count_nonzero(abertura)):>12} | "
            f"{int(np.count_nonzero(fechamento)):>14} | {len(aprovadas):>18} | {area_media_k:>15.1f}"
        )

        if codigo_forma == cv2.MORPH_RECT:
            paineis.append(("3. Abertura (MORPH_RECT 5x7)", abertura))
            anotada_principal = anotada
            aprovadas_principal = aprovadas

        nome_curto = nome_kernel.split(" ")[0]
        paineis.append((f"Fechamento - {nome_curto}", fechamento))

    paineis.append(("Componentes Filtrados (Pista)", anotada_principal))

    caminho_painel = BASE_DIR / "tp2-1b_morfologia.png"
    criar_painel_opencv(paineis, caminho_painel, largura=460, altura=320, colunas=4)

    larguras = [m["w"] for m in aprovadas_principal]
    alturas = [m["h"] for m in aprovadas_principal]
    areas = [m["area"] for m in aprovadas_principal]
    razoes = [m["aspect_ratio"] for m in aprovadas_principal]

    print("\n--- Estatisticas Detalhadas das Marcacoes de Pista (Kernel Retangular 5x7) ---")
    print(f"Total de componentes conexos brutos : {resultados_kernels['Retangular (MORPH_RECT)']['total_comp']}")
    print(f"Artefatos descartados pelo filtro   : {resultados_kernels['Retangular (MORPH_RECT)']['rejeitadas']}")
    print(f"Quantidade de marcacoes detectadas  : {len(aprovadas_principal)}")
    print(f"Largura media das marcacoes         : {float(np.mean(larguras)):.2f} px (DP = {float(np.std(larguras)):.2f} px)")
    print(f"Altura/Comprimento medio            : {float(np.mean(alturas)):.2f} px (DP = {float(np.std(alturas)):.2f} px)")
    print(f"Area media das marcacoes            : {float(np.mean(areas)):.2f} px² (min = {min(areas)}, max = {max(areas)})")
    print(f"Razao de aspecto media (h/w)        : {float(np.mean(razoes)):.2f}")

    # Analise comparativa dos elementos estruturantes (5x7):
    # - MORPH_RECT: preserva bordas retas e cantos ortogonais das faixas pintadas no asfalto,
    #   maximizando o preenchimento de falhas internas nas linhas tracejadas.
    # - MORPH_ELLIPSE: suaviza os contornos e remove ruido isotropico com menor distorcao angular,
    #   arredondando levemente as extremidades das faixas.
    # - MORPH_CROSS: atua apenas nas direcoes cardeais (horizontal/vertical), sendo mais conservador
    #   na remocao de ruido diagonal e retendo menos area nas bordas obliquas.
    print(f"\nPainel visual salvo em: {caminho_painel.name}")

    if not args.auto:
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.imshow(cv2.cvtColor(cv2.imread(str(caminho_painel)), cv2.COLOR_BGR2RGB))
        ax.set_title("TP2-1B: Pipeline Morfologico para Segmentacao de Marcacoes em BEV")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


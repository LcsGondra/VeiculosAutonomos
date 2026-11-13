import argparse
import time
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, carregar_imagem, normalizar_uint8


# Implementa filtragem espacial 2D manual em NumPy com borda reflect equivalente ao OpenCV
def convolucao_2d_manual_lacos(imagem_cinza, kernel_3x3):
    img_f64 = imagem_cinza.astype(np.float64)
    k_f64 = np.asarray(kernel_3x3, dtype=np.float64)
    h, w = img_f64.shape
    # mode='reflect' replica cv2.BORDER_REFLECT_101 sem duplicar o pixel de borda
    padded = np.pad(img_f64, ((1, 1), (1, 1)), mode="reflect")
    saida = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            janela = padded[y : y + 3, x : x + 3]
            saida[y, x] = np.sum(janela * k_f64)

    return saida


# Aplica combinacoes lineares de Sobel X/Y orientadas em 0, 45, 90 e 135 graus
def banco_filtros_direcionais_numpy(imagem_cinza):
    kx = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)
    ky = kx.T
    img_f64 = imagem_cinza.astype(np.float64)
    padded = np.pad(img_f64, ((1, 1), (1, 1)), mode="reflect")
    # sliding_window_view + einsum vetoriza a convolucao 2D em NumPy puro sem cv2.filter2D
    janelas = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))

    respostas = {}
    kernels = {}
    for angulo in (0, 45, 90, 135):
        theta = np.deg2rad(angulo)
        # Base direcional steerable: k(theta) = cos(theta)*Kx + sin(theta)*Ky
        k_dir = np.cos(theta) * kx + np.sin(theta) * ky
        kernels[angulo] = k_dir
        conv = np.einsum("ijkl,kl->ij", janelas, k_dir)
        respostas[angulo] = np.abs(conv)

    return respostas, kernels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bev_cinza = carregar_imagem("pista_bev.png", cinza=True)
    bordas_gt = carregar_imagem("bordas_gt.png", cinza=True) > 0

    kernel_sobel_x = np.array([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=np.float64)

    t0_man = time.perf_counter()
    sobel_manual = convolucao_2d_manual_lacos(bev_cinza, kernel_sobel_x)
    tempo_man_ms = (time.perf_counter() - t0_man) * 1000.0

    t0_cv = time.perf_counter()
    sobel_opencv = cv2.Sobel(bev_cinza, cv2.CV_64F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT_101)
    tempo_cv_ms = (time.perf_counter() - t0_cv) * 1000.0

    diferenca_abs = np.abs(sobel_manual - sobel_opencv)
    erro_max = float(np.max(diferenca_abs))
    erro_medio = float(np.mean(diferenca_abs))
    pixels_divergentes = int(np.count_nonzero(diferenca_abs > 1e-9))
    validado_allclose = bool(np.allclose(sobel_manual, sobel_opencv, atol=1e-9))

    respostas_dir, kernels_dir = banco_filtros_direcionais_numpy(bev_cinza)
    tensor_feature_maps = np.stack([respostas_dir[a] for a in (0, 45, 90, 135)], axis=-1).astype(np.float32)

    metricas_angulos = {}
    for angulo, mapa in respostas_dir.items():
        media_global = float(np.mean(mapa))
        media_bordas_faixa = float(np.mean(mapa[bordas_gt]))
        energia_bordas = float(np.sum(mapa[bordas_gt] ** 2))
        metricas_angulos[angulo] = {
            "media_global": media_global,
            "media_faixas": media_bordas_faixa,
            "energia_faixas": energia_bordas,
        }

    angulo_vencedor = max(metricas_angulos, key=lambda a: metricas_angulos[a]["media_faixas"])

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    paineis = [
        (bev_cinza, f"BEV Original ({bev_cinza.shape[1]}x{bev_cinza.shape[0]})"),
        (normalizar_uint8(sobel_manual), f"Sobel X Manual NumPy ({tempo_man_ms:.1f} ms)"),
        (normalizar_uint8(sobel_opencv), f"cv2.Sobel X ({tempo_cv_ms:.3f} ms)"),
        (normalizar_uint8(diferenca_abs), f"Erro Pixel a Pixel (Max={erro_max:.1e})"),
        (normalizar_uint8(respostas_dir[0]), f"Filtro 0° (Faixa={metricas_angulos[0]['media_faixas']:.1f})"),
        (normalizar_uint8(respostas_dir[45]), f"Filtro 45° (Faixa={metricas_angulos[45]['media_faixas']:.1f})"),
        (normalizar_uint8(respostas_dir[90]), f"Filtro 90° (Faixa={metricas_angulos[90]['media_faixas']:.1f})"),
        (normalizar_uint8(respostas_dir[135]), f"Filtro 135° (Faixa={metricas_angulos[135]['media_faixas']:.1f})"),
    ]

    for ax, (im, titulo) in zip(axes.flat, paineis):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.axis("off")

    fig.suptitle("TP2-2B: Convolucao 2D Manual (NumPy) vs cv2.Sobel e Banco de Filtros Direcionais", fontsize=13, fontweight="bold")
    fig.tight_layout()
    caminho_fig = BASE_DIR / "tp2-2b_convolucao_direcional.png"
    fig.savefig(caminho_fig, dpi=150, bbox_inches="tight")

    print("=== TP2 - Exercicio 2B: Convolucao 2D Manual e Banco de Filtros Direcionais ===")
    print(f"Resolucao da imagem BEV      : {bev_cinza.shape[1]}x{bev_cinza.shape[0]} px ({bev_cinza.size} pixels)")
    print(f"Tempo Sobel Manual (NumPy)   : {tempo_man_ms:.2f} ms")
    print(f"Tempo cv2.Sobel (C++/SIMD)   : {tempo_cv_ms:.4f} ms")
    print(f"Erro absoluto maximo (px)    : {erro_max:.12f}")
    print(f"Erro absoluto medio (MAE)    : {erro_medio:.12f}")
    print(f"Pixels divergentes (>1e-9)   : {pixels_divergentes} de {bev_cinza.size}")
    print(f"Validacao np.allclose        : {validado_allclose}")

    print("\n--- Resposta do Banco de Filtros Direcionais (0°, 45°, 90°, 135°) ---")
    print(f"{'Angulo do Filtro':<18} | {'Media Global':>14} | {'Media nas Bordas da Faixa':>26} | {'Energia nas Bordas':>20}")
    print("-" * 88)
    for angulo in (0, 45, 90, 135):
        m = metricas_angulos[angulo]
        destaque = " <-- MAIOR RESPOSTA" if angulo == angulo_vencedor else ""
        print(
            f"{str(angulo) + '°':<18} | {m['media_global']:>14.3f} | "
            f"{m['media_faixas']:>26.3f} | {m['energia_faixas']:>20.1f}{destaque}"
        )

    print(f"\nFormato do Tensor de Entrada : {bev_cinza.shape + (1,)} (H x W x C_in)")
    print(f"Formato do Tensor de Saida   : {tensor_feature_maps.shape} (H x W x C_out = 4 mapas de caracteristicas)")
    print(f"Angulo de maior resposta     : {angulo_vencedor}° (gradiente horizontal ortogonal as faixas longitudinais)")

    # Discussao (Filtros Direcionais vs Camadas Convolucionais de CNNs - Conexao DR2):
    # - No BEV, as faixas correm verticalmente (eixo Y), logo a transicao asfalto->tinta ocorre
    #   no eixo X, fazendo o filtro 0 graus (Sobel X) maximizar a resposta nas bordas longitudinais.
    # - O banco de 4 filtros 3x3 empilhados mapeia HxWx1 -> HxWx4, matematicamente equivalente a
    #   uma camada Conv2D(in_channels=1, out_channels=4, kernel_size=3, padding='reflect') sem bias.
    # - Diferentemente dos kernels analiticos fixos, CNNs (DR2) aprendem os filtros via backpropagation.
    print(f"\nFigura salva em: {caminho_fig.name}")

    if args.auto:
        plt.close("all")
    else:
        plt.show()


if __name__ == "__main__":
    main()


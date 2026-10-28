import os
import sys
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utils import imshow_keep_aspect_ratio

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "dados", "frontal_kitti.png")

img = cv2.imread(image_path)
if img is None:
    print(f"Erro: nao foi possivel abrir a imagem {image_path}")
    sys.exit(1)

h, w = img.shape[:2]
channels = img.shape[2] if img.ndim == 3 else 1
dtype = img.dtype
total_bytes = img.nbytes

print("=" * 60)
print("METADADOS DA IMAGEM VEICULAR FRONTAL (KITTI)")
print("=" * 60)
print(f"Resolucao: {w}x{h} (Largura x Altura)")
print(f"Tipo de Dado (dtype): {dtype}")
print(f"Numero de Canais: {channels}")
print(f"Tamanho em Memoria: {total_bytes} bytes ({total_bytes / (1024 * 1024):.2f} MB)")
print("=" * 60)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mid_y = h // 2
mid_x = w // 2

quadrants = {
    "Superior Esquerdo (Ceu / Cenario Distante)": gray[0:mid_y, 0:mid_x],
    "Superior Direito (Ceu / Margem Superior)": gray[0:mid_y, mid_x:w],
    "Inferior Esquerdo (Pista / Faixa Esquerda)": gray[mid_y:h, 0:mid_x],
    "Inferior Direito (Pista / Faixa Direita)": gray[mid_y:h, mid_x:w],
}

print("BRILHO MEDIO POR QUADRANTE (ESCALA 0-255):")
for name, quad in quadrants.items():
    mean_val = float(np.mean(quad))
    std_val = float(np.std(quad))
    print(f" - {name}: Media={mean_val:.2f} | Desvio Padrao={std_val:.2f}")
print("=" * 60)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_clahe = clahe.apply(gray)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l_chan, a_chan, b_chan = cv2.split(lab)
l_clahe = clahe.apply(l_chan)
color_clahe = cv2.cvtColor(cv2.merge([l_clahe, a_chan, b_chan]), cv2.COLOR_LAB2BGR)

hist_original = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
hist_clahe = cv2.calcHist([gray_clahe], [0], None, [256], [0, 256]).flatten()

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(hist_original, color="blue", label="Original (Antes CLAHE)", linewidth=1.5)
ax.plot(hist_clahe, color="red", label="Equalizada (Apos CLAHE)", linewidth=1.5)
ax.set_title("Comparacao de Histogramas de Intensidade: Original vs CLAHE")
ax.set_xlabel("Nivel de Cinza (0 - 255)")
ax.set_ylabel("Frequencia de Pixels")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper right")
plt.tight_layout()

hist_plot_path = os.path.join(base_dir, "dados", "histogramas_clahe.png")
plt.savefig(hist_plot_path, dpi=150)
plt.close(fig)
print(f"Grafico de histogramas salvo com sucesso em: {hist_plot_path}")

divider = np.full((h, 4, 3), (0, 255, 255), dtype=np.uint8)
side_by_side = np.hstack([img, divider, color_clahe])

cv2.putText(
    side_by_side,
    "ORIGINAL (KITTI)",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    4,
    cv2.LINE_AA,
)
cv2.putText(
    side_by_side,
    "ORIGINAL (KITTI)",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 255),
    2,
    cv2.LINE_AA,
)

cv2.putText(
    side_by_side,
    "CLAHE ADAPTATIVO (LAB L-CHANNEL)",
    (w + 34, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 0, 0),
    4,
    cv2.LINE_AA,
)
cv2.putText(
    side_by_side,
    "CLAHE ADAPTATIVO (LAB L-CHANNEL)",
    (w + 34, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (0, 255, 255),
    2,
    cv2.LINE_AA,
)

preview_path = os.path.join(base_dir, "dados", "original_vs_clahe.png")
cv2.imwrite(preview_path, side_by_side)
print(f"Imagem comparativa lado a lado salva em: {preview_path}")

print("=" * 60)
print("ANALISE TECNICA: IMPACTO DE ILUMINACAO NA PERCEPCAO VEICULAR")
print("1. Variacoes bruscas de iluminacao (tuneis, viadutos, contraluz) provocam perda")
print("   de contraste local entre a marcacao da pista e o asfalto.")
print("2. A equalizacao global (cv2.equalizeHist) tende a super-amplificar o ruido no asfalto")
print("   ou estourar o ceu. O CLAHE resolve isso dividindo a imagem em grades (tiles 8x8)")
print("   e limitando a amplificacao de contraste (clipLimit=2.0).")
print("3. Como demonstrado no histograma, os pixels das faixas que estavam comprimidos na regiao")
print("   escura passam a ser redistribuidos, facilitando a deteccao por gradiente Sobel e Canny.")
print("=" * 60)

if "--auto" not in sys.argv:
    win_title = "TP1 - Ex 1A: Original vs CLAHE Adaptativo"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1280, 480)
    while True:
        imshow_keep_aspect_ratio(win_title, side_by_side)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break
    cv2.destroyAllWindows()

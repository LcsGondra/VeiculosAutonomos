import os
import sys
import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio

base_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(base_dir, "dados", "frontal_kitti.png")
grid_out_path = os.path.join(base_dir, "dados", "transformacoes_afins_4a.png")

img = cv2.imread(image_path)
if img is None:
    print(f"Erro: nao foi possivel abrir a imagem {image_path}")
    sys.exit(1)

h, w = img.shape[:2]
center = (w / 2.0, h / 2.0)

print("=" * 65)
print("TRANSFORMACOES GEOMETRICAS AFINS 2D E COMPOSICAO MATRICIAL")
print(f"Imagem de Entrada: frontal_kitti.png ({w}x{h} px)")
print("=" * 65)

tx, ty = 80.0, -40.0
M_translacao = np.float32([[1.0, 0.0, tx], [0.0, 1.0, ty]])
img_translacao = cv2.warpAffine(img, M_translacao, (w, h), borderValue=(30, 30, 30))

print("\n1. TRANSLACAO 2D (tx=+80 px, ty=-40 px):")
print(f"Matriz de Translacao (2x3):\n{M_translacao}")

ang_deg = 15.0
M_rotacao = cv2.getRotationMatrix2D(center, ang_deg, scale=1.0)
img_rotacao = cv2.warpAffine(img, M_rotacao, (w, h), borderValue=(30, 30, 30))

print(f"\n2. ROTACAO EM TORNO DO CENTRO ({ang_deg} graus anti-horario):")
print(f"Matriz de Rotacao (2x3):\n{M_rotacao}")

scale_factor = 1.30
M_escala = cv2.getRotationMatrix2D(center, 0.0, scale=scale_factor)
img_escala = cv2.warpAffine(img, M_escala, (w, h), borderValue=(30, 30, 30))

print(f"\n3. ESCALA / ZOOM DIGITAL ({scale_factor * 100:.0f}% em relacao ao centro):")
print(f"Matriz de Escala (2x3):\n{M_escala}")

src_tri = np.float32([[0, 0], [w - 1, 0], [0, h - 1]])
dst_tri = np.float32([[w * 0.05, h * 0.10], [w * 0.90, h * 0.05], [w * 0.10, h * 0.95]])
M_afim = cv2.getAffineTransform(src_tri, dst_tri)
img_afim = cv2.warpAffine(img, M_afim, (w, h), borderValue=(30, 30, 30))

print("\n4. TRANSFORMACAO AFIM CUSTOMIZADA (3 PONTOS / CISALHAMENTO):")
print(f"Matriz Afim Customizada (2x3):\n{M_afim}")

M_rot_homog = np.vstack([M_rotacao, [0.0, 0.0, 1.0]])
M_trans_homog = np.vstack([M_translacao, [0.0, 0.0, 1.0]])
M_comp_homog = np.dot(M_trans_homog, M_rot_homog)
M_comp_2x3 = M_comp_homog[0:2, :]
img_composta = cv2.warpAffine(img, M_comp_2x3, (w, h), borderValue=(30, 30, 30))

print("\n5. DEMONSTRACAO DE COMPOSICAO POR MULTIPLICACAO MATRICIAL:")
print(f"Matriz Homogenea de Rotacao (3x3):\n{M_rot_homog}")
print(f"Matriz Homogenea de Translacao (3x3):\n{M_trans_homog}")
print(f"Produto Matricial M_trans x M_rot (3x3):\n{M_comp_homog}")
print(f"Matriz Composta Final Aplicada com cv2.warpAffine (2x3):\n{M_comp_2x3}")

thumb_w, thumb_h = 480, 200


def prepare_thumb(image, label):
    resized = cv2.resize(image, (thumb_w, thumb_h))
    cv2.putText(resized, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(resized, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    return resized


t_orig = prepare_thumb(img, "Original (KITTI)")
t_trans = prepare_thumb(img_translacao, "1. Translacao (tx=+80, ty=-40)")
t_rot = prepare_thumb(img_rotacao, "2. Rotacao (+15 deg)")
t_esc = prepare_thumb(img_escala, "3. Escala / Zoom (130%)")
t_afim = prepare_thumb(img_afim, "4. Afim Customizada (Shear)")
t_comp = prepare_thumb(img_composta, "5. Composta (Rotacao + Translacao)")

row1 = np.hstack([t_orig, t_trans])
row2 = np.hstack([t_rot, t_esc])
row3 = np.hstack([t_afim, t_comp])
grid = np.vstack([row1, row2, row3])

cv2.imwrite(grid_out_path, grid)
print("=" * 65)
print(f"Painel consolidado das 6 imagens salvo com sucesso em: {grid_out_path}")
print("=" * 65)

print("\nUTILIDADE NO PRE-PROCESSAMENTO PARA REDES NEURAIS DE DIRECAO AUTONOMA:")
print("1. Translacao: Simula mudancas na posicao lateral e longitudinal do veiculo")
print("   na faixa, prevenindo overfitting da rede a trajetorias estritamente centradas.")
print("2. Rotacao: Simula variacoes no angulo de atitude (yaw/pitch) da camera causadas")
print("   por irregularidades no pavimento, curvas fechadas ou balanco da suspensao.")
print("3. Escala (Zoom Digital): Simula variacoes na distancia focal e aproxima/afasta")
print("   obstaculos e marcacoes de pista, aumentando a robustez para diferentes cameras.")
print("4. Transformacao Afim e Composicao: Permite criar augmentations realistas e continuas")
print("   com um unico passo computacional (warpAffine), aplicando correcoes de inclinacao")
print("   de montagem de camera (camera calibration/rectification).")
print("=" * 65)

if "--auto" not in sys.argv:
    win_title = "TP1 - Ex 4A: Transformacoes Geometricas 2D e Composicao"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1100, 700)
    while True:
        imshow_keep_aspect_ratio(win_title, grid)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break
    cv2.destroyAllWindows()

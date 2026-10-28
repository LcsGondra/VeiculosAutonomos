import csv
import os
import random
import sys
import cv2
import numpy as np
from utils import imshow_keep_aspect_ratio, letterbox_image

base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "dados", "direcao_udacity.mp4")
out_dir = os.path.join(base_dir, "dados", "dataset_normalizado")
grid_path = os.path.join(base_dir, "dados", "grid_dataset_224.png")
csv_info_path = os.path.join(out_dir, "dataset_info.csv")

os.makedirs(out_dir, exist_ok=True)

random.seed(42)
np.random.seed(42)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Erro: nao foi possivel abrir o video {video_path}")
    sys.exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_indices = np.linspace(5, total_frames - 10, 20, dtype=int)

raw_samples = []
for idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    if ret and frame is not None:
        raw_samples.append((idx, frame))
cap.release()

if len(raw_samples) < 20:
    print(f"Erro: foram obtidas apenas {len(raw_samples)} imagens.")
    sys.exit(1)

print("=" * 65)
print("PIPELINE DE NORMALIZACAO DE DATASET PARA CNNS VEICULARES")
print("Total de Amostras Carregadas: 20 frames de direcao")
print("Resolucao Alvo Padronizada: 224x224 com Letterbox")
print("=" * 65)

processed_for_grid = []
dataset_records = []

for i, (orig_frame_idx, raw_img) in enumerate(raw_samples):
    base_steering = 0.12 * np.sin(i * 0.45)
    brightness_factor = float(np.random.uniform(0.80, 1.25))
    varied_raw = np.clip(raw_img.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)

    img_letter, scale, offset = letterbox_image(varied_raw, target_size=(224, 224), fill_color=(114, 114, 114))

    do_flip = (i % 2 == 1)
    if do_flip:
        aug_img = cv2.flip(img_letter, 1)
        aug_steering = -base_steering
    else:
        aug_img = img_letter.copy()
        aug_steering = base_steering

    rot_angle = float(np.random.uniform(-10.0, 10.0))
    zoom_scale = float(np.random.uniform(0.90, 1.10))

    M_aug = cv2.getRotationMatrix2D((112.0, 112.0), rot_angle, zoom_scale)
    aug_img = cv2.warpAffine(aug_img, M_aug, (224, 224), borderValue=(114, 114, 114))

    img_float = aug_img.astype(np.float32) / 255.0
    mean_ch = np.mean(img_float, axis=(0, 1))
    std_ch = np.std(img_float, axis=(0, 1)) + 1e-7
    zscore_img = (img_float - mean_ch) / std_ch
    norm_mean = np.mean(zscore_img, axis=(0, 1))
    norm_std = np.std(zscore_img, axis=(0, 1))

    sample_filename = f"sample_{i+1:02d}.png"
    sample_file_path = os.path.join(out_dir, sample_filename)
    cv2.imwrite(sample_file_path, aug_img)

    dataset_records.append(
        {
            "id": i + 1,
            "orig_frame": orig_frame_idx,
            "filename": sample_filename,
            "brightness_factor": brightness_factor,
            "base_steering": base_steering,
            "aug_steering": aug_steering,
            "flipped": do_flip,
            "rot_deg": rot_angle,
            "zoom": zoom_scale,
            "mean_b": float(mean_ch[0]),
            "mean_g": float(mean_ch[1]),
            "mean_r": float(mean_ch[2]),
            "std_b": float(std_ch[0]),
            "std_g": float(std_ch[1]),
            "std_r": float(std_ch[2]),
            "zscore_mean_abs": float(np.mean(np.abs(norm_mean))),
            "zscore_std_mean": float(np.mean(norm_std)),
        }
    )

    cell = aug_img.copy()
    cv2.putText(cell, f"#{i+1:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(cell, f"#{i+1:02d}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

    steer_text = f"Str: {aug_steering:+.2f}"
    cv2.putText(cell, steer_text, (8, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(cell, steer_text, (8, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    processed_for_grid.append(cell)

with open(csv_info_path, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(dataset_records[0].keys()))
    writer.writeheader()
    writer.writerows(dataset_records)

rows = []
for r in range(4):
    row_imgs = processed_for_grid[r * 5 : (r + 1) * 5]
    rows.append(np.hstack(row_imgs))
grid_img = np.vstack(rows)

cv2.imwrite(grid_path, grid_img)

print(f"Dataset de 20 amostras persistido com sucesso na pasta: {out_dir}")
print(f"Arquivo CSV com metadados e estatisticas de cada amostra: {csv_info_path}")
print(f"Grid 4x5 consolidado ({grid_img.shape[1]}x{grid_img.shape[0]} px) salvo em: {grid_path}")
print("=" * 65)
print("JUSTIFICATIVAS TECNICAS PARA REDES NEURAIS (CNNS) DE DIRECAO AUTONOMA:")
print("1. Letterbox (224x224 com Aspect Ratio Preservado):")
print("   - Redes neurais convolucionais (ex: ResNet, MobileNet, PilotNet) exigem tensores")
print("     de entrada de dimensao fixa. Redimensionamentos anamorficos (esticar a imagem)")
print("     distorcem a geometria das faixas e a relacao angular do ponto de fuga.")
print("   - O padding neutro (cinza 114) mantem as proporcoes fisicas reais da pista.")
print("2. Padronizacao por Canal (Z-score: subtracao da media e divisao pelo desvio):")
print("   - Centraliza o espaco de caracteristicas em zero com variancia unitaria.")
print("   - Acelera dramaticamente a convergencia do gradiente descendente (SGD/Adam),")
print("     evita saturacao de funcoes de ativacao e atenua variacoes globais de exposicao.")
print("3. Data Augmentation Geometrico (Flip com Inversao de Steering):")
print("   - Em pistas reais, ha forte vies estatistico para retas ou curvas em um unico sentido.")
print("   - O espelhamento horizontal duplica o dataset e garante simetria de distribuicao")
print("     angular, condicionando a inversao identica do sinal do angulo de direcao.")
print("4. Rotacao (+-10 graus) e Zoom (+-10%):")
print("   - Simula oscilacoes dinamicas de rolagem/arfagem do veiculo e tolerancias de montagem.")
print("=" * 65)

if "--auto" not in sys.argv:
    win_title = "TP1 - Ex 4B: Grid 4x5 de Dataset Normalizado (224x224)"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, 1120, 896)
    while True:
        imshow_keep_aspect_ratio(win_title, grid_img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q") or key == ord("Q") or key == 27:
            break
    cv2.destroyAllWindows()

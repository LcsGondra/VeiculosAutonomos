from pathlib import Path
import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DADOS_DIR = BASE_DIR / "dados"


def carregar_imagem(nome, cinza=False):
    caminho = DADOS_DIR / nome
    flag = cv2.IMREAD_GRAYSCALE if cinza else cv2.IMREAD_COLOR
    imagem = cv2.imread(str(caminho), flag)
    if imagem is None:
        raise FileNotFoundError(f"Imagem nao encontrada: {caminho}")
    return imagem


def normalizar_uint8(matriz):
    arr = np.asarray(matriz, dtype=np.float64)
    max_val = float(np.max(np.abs(arr)))
    if max_val <= 1e-12:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.uint8(np.clip((np.abs(arr) / max_val) * 255.0, 0, 255))


def criar_painel_opencv(pares, caminho_saida, largura=540, altura=310, colunas=3):
    quadros = []
    for legenda, imagem in pares:
        img = imagem.copy()
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_redim = cv2.resize(img, (largura, altura), interpolation=cv2.INTER_AREA)
        cv2.rectangle(img_redim, (0, 0), (largura, 40), (22, 31, 46), -1)
        cv2.putText(
            img_redim,
            str(legenda)[:46],
            (12, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        quadros.append(img_redim)

    ncol = min(colunas, len(quadros))
    while len(quadros) % ncol != 0:
        quadros.append(np.full_like(quadros[0], (235, 235, 235)))

    linhas = [np.hstack(quadros[i : i + ncol]) for i in range(0, len(quadros), ncol)]
    resultado = np.vstack(linhas)
    cv2.imwrite(str(caminho_saida), resultado)
    return resultado


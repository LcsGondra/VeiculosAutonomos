import argparse
import os
import cv2
import matplotlib
import numpy as np
from utils import BASE_DIR, carregar_imagem, criar_painel_opencv


def detectar_orb_e_casar(img1, img2, nfeatures=2500, ratio_lowe=0.80):
    cinza1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    cinza2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=12)
    kp1, desc1 = orb.detectAndCompute(cinza1, None)
    kp2, desc2 = orb.detectAndCompute(cinza2, None)

    if desc1 is None or desc2 is None:
        raise RuntimeError("Descritores ORB nao encontrados nas imagens.")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pares_knn = bf.knnMatch(desc2, desc1, k=2)

    bons_matches = []
    for par in pares_knn:
        if len(par) == 2:
            m, n = par
            if m.distance < ratio_lowe * n.distance:
                bons_matches.append(m)

    return kp1, desc1, kp2, desc2, pares_knn, bons_matches


def estimar_homografia_ransac(kp1, kp2, bons_matches, limiar_ransac=3.0):
    if len(bons_matches) < 4:
        raise RuntimeError(f"Matches insuficientes ({len(bons_matches)}) para homografia.")

    pts_src = np.float32([kp2[m.queryIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
    pts_dst = np.float32([kp1[m.trainIdx].pt for m in bons_matches]).reshape(-1, 1, 2)

    H, mascara = cv2.findHomography(pts_src, pts_dst, cv2.RANSAC, limiar_ransac)
    if H is None or mascara is None:
        raise RuntimeError("Falha ao estimar homografia via RANSAC.")

    inliers_mask = mascara.ravel().astype(bool)
    pts_proj = cv2.perspectiveTransform(pts_src, H)
    erros = np.linalg.norm((pts_proj - pts_dst).reshape(-1, 2), axis=1)
    erro_medio_inliers = float(np.mean(erros[inliers_mask])) if np.any(inliers_mask) else float("nan")
    erro_max_inliers = float(np.max(erros[inliers_mask])) if np.any(inliers_mask) else float("nan")
    erro_medio_todos = float(np.mean(erros))

    return H, inliers_mask, erros, erro_medio_inliers, erro_max_inliers, erro_medio_todos


def construir_mosaico_panoramico(img1, img2, H):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    cantos1 = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2)
    cantos2 = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    cantos2_proj = cv2.perspectiveTransform(cantos2, H).reshape(-1, 2)

    todos_cantos = np.vstack([cantos1.reshape(-1, 2), cantos2_proj])
    xmin, ymin = np.floor(todos_cantos.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(todos_cantos.max(axis=0)).astype(int)

    T = np.array([[1.0, 0.0, -xmin], [0.0, 1.0, -ymin], [0.0, 0.0, 1.0]], dtype=np.float64)
    largura_canvas = int(xmax - xmin)
    altura_canvas = int(ymax - ymin)
    tamanho_canvas = (largura_canvas, altura_canvas)

    warp1 = cv2.warpPerspective(img1, T, tamanho_canvas)
    warp2 = cv2.warpPerspective(img2, T @ H, tamanho_canvas)

    mask1 = cv2.warpPerspective(np.full((h1, w1), 255, dtype=np.uint8), T, tamanho_canvas) > 0
    mask2 = cv2.warpPerspective(np.full((h2, w2), 255, dtype=np.uint8), T @ H, tamanho_canvas) > 0

    mosaico = warp1.copy()
    somente_vista2 = mask2 & ~mask1
    mosaico[somente_vista2] = warp2[somente_vista2]

    intersecao = mask1 & mask2
    mosaico[intersecao] = np.clip(
        0.5 * warp1[intersecao].astype(np.float32) + 0.5 * warp2[intersecao].astype(np.float32),
        0,
        255,
    ).astype(np.uint8)

    return mosaico, warp1, warp2, intersecao


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()

    if args.auto:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    visao1 = carregar_imagem("visao_01.png")
    visao2 = carregar_imagem("visao_02.png")

    kp1, desc1, kp2, desc2, pares_knn, bons_matches = detectar_orb_e_casar(
        visao1, visao2, nfeatures=2500, ratio_lowe=0.80
    )
    H, inliers_mask, erros, erro_inliers, erro_max_inliers, erro_todos = estimar_homografia_ransac(
        kp1, kp2, bons_matches, limiar_ransac=3.0
    )
    mosaico, warp1, warp2, intersecao = construir_mosaico_panoramico(visao1, visao2, H)

    inliers_matches = [m for m, ok in zip(bons_matches, inliers_mask) if ok]
    img_matches = cv2.drawMatches(
        visao2,
        kp2,
        visao1,
        kp1,
        inliers_matches[:80],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 100, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    caminho_matches = BASE_DIR / "tp2-1a_matches.png"
    caminho_mosaico = BASE_DIR / "tp2-1a_mosaico.png"
    cv2.imwrite(str(caminho_matches), img_matches)

    criar_painel_opencv(
        [
            ("Vista 01 (Camera Frontal-Esq)", visao1),
            ("Vista 02 (Camera Frontal-Dir)", visao2),
            ("Casamento ORB + Inliers RANSAC", img_matches),
            ("Mosaico Panoramico Alinhado", mosaico),
        ],
        caminho_mosaico,
        largura=580,
        altura=320,
        colunas=2,
    )

    num_inliers = int(np.sum(inliers_mask))
    num_outliers = int(len(bons_matches) - num_inliers)
    taxa_inliers = (num_inliers / max(len(bons_matches), 1)) * 100.0
    frac_sobreposicao = float(np.sum(intersecao)) / float( np.sum((warp1.sum(axis=2) > 0) | (warp2.sum(axis=2) > 0)) ) * 100.0

    print("=== TP2 - Exercicio 1A: Alinhamento Multi-Camera por Homografia ===")
    print(f"Resolucao Vista 01          : {visao1.shape[1]}x{visao1.shape[0]} px")
    print(f"Resolucao Vista 02          : {visao2.shape[1]}x{visao2.shape[0]} px")
    print(f"Keypoints ORB detectados    : Vista 01 = {len(kp1)} | Vista 02 = {len(kp2)}")
    print(f"Pares brutos BFMatcher (k=2): {len(pares_knn)}")
    print(f"Matches apos filtro de Lowe : {len(bons_matches)} (ratio = 0.80)")
    print(f"Inliers RANSAC (limiar 3px) : {num_inliers} de {len(bons_matches)} ({taxa_inliers:.2f}%) | Outliers = {num_outliers}")
    print(f"Erro medio reprojecao inlier: {erro_inliers:.4f} px (max = {erro_max_inliers:.4f} px)")
    print(f"Erro medio global (c/ out.) : {erro_todos:.4f} px")
    print(f"Resolucao Mosaico Panoramico: {mosaico.shape[1]}x{mosaico.shape[0]} px (sobreposicao util = {frac_sobreposicao:.1f}%)")
    print("\nMatriz de Homografia H (Vista 02 -> Vista 01):")
    for linha in H:
        print(f"  [{linha[0]:12.6f}  {linha[1]:12.6f}  {linha[2]:12.6f}]")

    print("\n--- Aplicacao de Homografia Multi-Camera em Sistemas Surround View 360° ---")
    print("1. Geometria Planar: Em sistemas Surround View (AVM - Around View Monitor), 4 a 6 cameras fisheye")
    print("   (frontal, traseira e laterais nos retrovisores) capturam vistas com sobreposicao parcial nos")
    print("   4 cantos do veiculo. Apos a correcao de distorcao radial/tangencial (intrinsecos), cada vista")
    print("   e projetada no plano da via (z = 0) usando uma homografia H_i (Inverse Perspective Mapping).")
    print("2. Costura e Alinhamento: Homografias inter-cameras alinham as zonas de intersecao no referencial")
    print("   comum Bird's Eye View (BEV) centrado no chassi. O blending ponderado nas bordas de transicao")
    print("   suaviza diferencas de exposicao automatica (AE) e balanco de branco (AWB) entre os sensores.")
    print("3. Limitacao Fisica (Paralaxe): Como a homografia assume cena perfeitamente planar ou rotacao pura")
    print("   sem translacao entre centros opticos, objetos tridimensionais acima do solo (pedestres, meios-fios,")
    print("   carros proximos) sofrem distorcao radial e fantasma (ghosting) nas costuras do mosaico 360°.")
    print(f"\nImagens salvas em: {caminho_mosaico.name} e {caminho_matches.name}")

    if not args.auto:
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.imshow(cv2.cvtColor(cv2.imread(str(caminho_mosaico)), cv2.COLOR_BGR2RGB))
        ax.set_title("TP2-1A: Alinhamento Multi-Camera por Homografia")
        ax.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()


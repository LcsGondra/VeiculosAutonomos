# Relatório Técnico — TP2 de Veículos Autônomos (DR4)

- **Instituição:** Instituto Infnet — Escola Superior de Tecnologia — Graduação em Engenharia de Computação
- **Disciplina:** Veículos Autônomos (DR4)
- **Trabalho Prático:** TP 02 — Homografia, Morfologia, Filtragem Espacial, Formas/Keypoints e Simulação MetaDrive + YOLOv8n
- **Aluno:** Lucas Dias de Gondra (`lucas.gondra@al.infnet.edu.br`)
- **Repositório Git:** https://github.com/LcsGondra/VeiculosAutonomos/tree/main/tp2

---

## Preparação e Execução no Terminal (Windows PowerShell)

```powershell
cd c:\Users\Lucas\Documents\codes\BlocoRobotica\VeiculosAutonomos\tp2
pip install -r requirements.txt
python tp2-1a.py
python tp2-1b.py
python tp2-2a.py
python tp2-2b.py
python tp2-3a.py
python tp2-3b.py
python tp2-4a.py
python tp2-4b.py
```

---

## 1. Exercício 1: Homografia e Operações Morfológicas

### 1.1 Item A: Alinhamento Multi-Câmera por Homografia (`tp2-1a.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-1a.py
- **Comando para executar:** `python tp2-1a.py`

#### Resultados Quantitativos
- **Resolução das Vistas:** `visao_01.png` (`850x370 px`) e `visao_02.png` (`850x370 px`).
- **Detecção ORB:** `2500` keypoints na Vista 01 e `2500` keypoints na Vista 02 (`fastThreshold=12`).
- **Casamento BFMatcher + Filtro de Razão de Lowe (`ratio=0.80`):** `1684` correspondências aprovadas de `2500` pares brutos (`k=2`, norma de Hamming).
- **Estimativa RANSAC (`limiar = 3.0 px`):** **`1634` inliers** de `1684` matches (**`97.03%`** de consenso geométrico) e `50` outliers rejeitados.
- **Erro de Reprojeção Médio (Inliers):** **`0.3383 px`** (máximo entre inliers = `3.4461 px`; erro global incluindo outliers = `5.3830 px`).
- **Mosaico Panorâmico (`cv2.warpPerspective`):** Resolução final de **`1180x372 px`**, com área de sobreposição útil de **`44.1%`** combinada via blending linear ponderado.

$$
H_{2 \to 1} = \begin{bmatrix} 1.000510 & -0.000125 & 329.930885 \\ 0.000111 & 1.000160 & -0.038441 \\ 0.000001 & -0.000001 & 1.000000 \end{bmatrix}
$$

#### Aplicação em Sistemas de Visão 360° (Surround View / AVM)
1. **Projeção Planar ao Solo:** Em sistemas automotivos *Around View Monitor* (AVM), 4 a 6 câmeras grande-angulares (*fisheye*) instaladas na grade frontal, tampa traseira e espelhos retrovisores laterais capturam o entorno do veículo com zonas de interseção nos 4 cantos. Após compensar a distorção radial/tangencial intrínseca, cada câmera aplica uma matriz de homografia $H_i$ (*Inverse Perspective Mapping*) que projeta sua imagem para o plano horizontal do asfalto ($Z = 0$) num referencial Bird's Eye View (BEV) unificado sobre o chassi.
2. **Alinhamento e Costura (*Stitching*):** Homografias entre câmeras adjacentes garantem continuidade geométrica nas faixas de transição, onde máscaras de *alpha blending* suavizam diferenças fotométricas causadas por controle automático de ganho/exposição (AE) e balanço de branco (AWB) independente entre sensores.
3. **Limitação Física de Paralaxe:** Como a homografia modela exatamente apenas cenas planares ($Z=0$) ou rotações puras em torno do mesmo centro óptico, obstáculos tridimensionais elevados (pedestres, meios-fios, veículos vizinhos) violam a hipótese planar e sofrem alongamento radial e duplicação fantasma (*ghosting*) nas costuras do mosaico 360°.

---

### 1.2 Item B: Pipeline Morfológico para Segmentação de Marcações em BEV (`tp2-1b.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-1b.py
- **Comando para executar:** `python tp2-1b.py`

#### Comparação dos 3 Elementos Estruturantes (`5x7`)

| Elemento Estruturante | Pixels Após Abertura | Pixels Após Fechamento | Marcações Isoladas | Área Média (`px²`) |
| :--- | :---: | :---: | :---: | :---: |
| **Retangular (`MORPH_RECT`)** | `13.145` | `14.268` | `7` | `2026.3` |
| **Elíptico (`MORPH_ELLIPSE`)** | `13.196` | `13.810` | `7` | `1960.9` |
| **Cruz (`MORPH_CROSS`)** | `12.971` | `13.515` | `7` | `1919.6` |

#### Estatísticas de Componentes Conexos (`cv2.connectedComponentsWithStats`)
- **Filtro Aplicado:** `area >= 100 px²` e razão de aspecto `h / w >= 1.5`.
- **Componentes Conexos Brutos:** `8` (`1` artefato residual descartado e **`7` marcações de pista aprovadas**).
- **Dimensões Médias das Marcações Detectadas (`MORPH_RECT`):**
  - **Largura média:** **`19.14 px`** ($\sigma = 2.53\text{ px}$)
  - **Altura/Comprimento médio:** **`134.00 px`** ($\sigma = 143.19\text{ px}$, englobando tracejados curtos de `65 px` e a linha contínua direita de `480 px`)
  - **Área média:** **`2026.29 px²`** (mín = `1023 px²`, máx = `7245 px²`)
  - **Razão de aspecto média ($h/w$):** **`7.05`**

---

## 2. Exercício 2: Filtragem Espacial e Detecção de Bordas

### 2.1 Item A: Comparação de 4 Operadores de Detecção de Bordas (`tp2-2a.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-2a.py
- **Comando para executar:** `python tp2-2a.py`

#### Tabela 1 — Cena Frontal Veicular Real (`frontal_kitti.png`, `1224x370 px`)

| Operador | Pixels Ativados | Taxa Ativa (%) | Tempo Mediano (`ms`) |
| :--- | :---: | :---: | :---: |
| **Sobel (X+Y Magnitude)** | `124.178` | `27.42%` | `19.227` |
| **LoG ($\sigma=1.2$, $5\times 5$)** | `74.932` | `16.55%` | `7.075` |
| **Canny (`50 / 110`)** | `88.978` | `19.65%` | `2.288` |
| **Canny (`100 / 200`)** | `57.888` | `12.78%` | `1.220` |
| **DoG ($\sigma_1=1.0, \sigma_2=2.2$)** | `117.974` | `26.05%` | `7.833` |

#### Tabela 2 — Benchmark sob Condições de Sombra e Asfalto Molhado (`pista_bev.png`)

| Cenário | Operador | Pixels Ativos | Tempo (`ms`) | Precisão | Recall | F1-Score |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Normal** | Sobel (X+Y Mag) | `6.280` | `12.293` | `0.744` | `1.000` | `0.853` |
| **Normal** | LoG ($\sigma=1.2$) | `18.171` | `3.751` | `0.623` | `1.000` | `0.768` |
| **Normal** | Canny (`50/110`) | `3.048` | `0.369` | `0.738` | `1.000` | `0.849` |
| **Normal** | Canny (`100/200`) | `3.048` | `0.311` | `0.738` | `1.000` | `0.849` |
| **Normal** | DoG (`1.0 vs 2.2`) | `22.954` | `5.483` | `0.494` | `1.000` | `0.661` |
| **Sombra** | Sobel (X+Y Mag) | `8.758` | `10.117` | `0.542` | `1.000` | `0.703` |
| **Sombra** | LoG ($\sigma=1.2$) | `15.200` | `3.850` | `0.636` | `1.000` | **`0.778`** |
| **Sombra** | Canny (`50/110`) | `4.300` | `0.787` | `0.530` | `1.000` | `0.693` |
| **Sombra** | Canny (`100/200`) | `4.300` | `0.487` | `0.530` | `1.000` | `0.693` |
| **Sombra** | DoG (`1.0 vs 2.2`) | `21.629` | `5.512` | `0.493` | `1.000` | `0.660` |
| **Asfalto Molhado** | Sobel (X+Y Mag) | `8.741` | `10.731` | `0.545` | `1.000` | `0.706` |
| **Asfalto Molhado** | LoG ($\sigma=1.2$) | `11.861` | `4.113` | `0.779` | `1.000` | **`0.876`** |
| **Asfalto Molhado** | Canny (`50/110`) | `4.559` | `0.269` | `0.506` | `1.000` | `0.672` |
| **Asfalto Molhado** | Canny (`100/200`) | `4.506` | `0.215` | `0.509` | `1.000` | `0.674` |
| **Asfalto Molhado** | DoG (`1.0 vs 2.2`) | `18.041` | `5.130` | `0.608` | `1.000` | `0.756` |

#### Justificativa de Robustez (Asfalto Molhado e Sombras)
- No dataset sintético com métrica tolerante a 2 px, o **LoG ($\sigma=1.2$)** atingiu o maior F1 numérico em **Sombra (`0.778`)** e **Asfalto Molhado (`0.876`)** porque sua suavização Gaussiana prévia atenua o ruído aditivo enquanto a borda dupla do Laplaciano preenche a vizinhança de tolerância da máscara de referência.
- Contudo, em **pipelines embarcados reais** (onde se exige resposta de 1 px de espessura para a Transformada de Hough e latência sub-milissegundo), o **Canny (`100/200`) precedido por filtro Gaussiano** é a escolha de engenharia mais robusta (`0.215 ms` a `1.220 ms`): a **supressão não-máxima (NMS)** evita bordas espessas, o limiar superior (`200`) rejeita reflexos especulares fracos na lâmina d'água e a **histerese** reconecta os trechos de faixa atenuados pela sombra sem disparar falsos positivos na textura do pavimento.

---

### 2.2 Item B: Convolução 2D Manual em NumPy e Banco de Filtros Direcionais (`tp2-2b.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-2b.py
- **Comando para executar:** `python tp2-2b.py`

#### Validação Pixel a Pixel (`convolucao_2d_manual_lacos` vs `cv2.Sobel`)
- **Imagem avaliada:** `pista_bev.png` (`640x400 px` = `256.000` pixels) com padding `reflect`.
- **Erro absoluto máximo:** **`0.000000000000`** (`MAE = 0.0`, **`0` pixels divergentes** de `256.000`, `np.allclose = True`).
- **Tempo de execução:** `512.00 ms` em laços explícitos Python/NumPy vs `0.6155 ms` em `cv2.Sobel` (C++/SIMD).

#### Resposta do Banco de Filtros Direcionais (`0°, 45°, 90°, 135°`)

| Ângulo do Filtro | Média Global | Média nas Bordas da Faixa | Energia nas Bordas |
| :--- | :---: | :---: | :---: |
| **0° (Gradiente Horizontal)** | **`17.843`** | **`643.441`** | **`1.022.064.461,0`** |
| **45° (Diagonal Principal)** | `14.411` | `496.858` | `566.788.656,0` |
| **90° (Gradiente Vertical)** | `5.160` | `91.671` | `123.447.857,0` |
| **135° (Diagonal Secundária)** | `14.489` | `502.151` | `578.723.662,0` |

#### Discussão: Filtros Direcionais e Camadas Convolucionais de CNNs (Conexão DR2)
1. **Ângulo de Maior Resposta (`0°`):** Na projeção BEV, as faixas de rolamento são estruturas longitudinais alinhadas ao eixo vertical $Y$. A transição de intensidade entre o asfalto escuro e a tinta clara ocorre perpendicularmente à faixa (ao longo do eixo horizontal $X$). Logo, o filtro direcional de **`0°` (`Sobel X`)** maximiza a resposta média nas bordas (`643.441`), enquanto o filtro de **`90°` (`Sobel Y`)** responde apenas nas extremidades horizontais das marcações tracejadas (`91.671`).
2. **Relação com Camadas `Conv2D` (DR2):** O empilhamento dos 4 mapas de resposta transforma a imagem de entrada $(400, 640, 1)$ num tensor $(400, 640, 4)$, operação idêntica a uma camada `Conv2D(in_channels=1, out_channels=4, kernel_size=3)`. Enquanto no banco clássico os coeficientes são fixados analiticamente pelo engenheiro, nas CNNs estudadas em DR2 os pesos dos filtros das primeiras camadas convergem via *backpropagation* para detectores direcionais de bordas e frequências (análogos a Sobel/Gabor), cujos *feature maps* são recombinados nas camadas profundas para reconhecer semântica complexa (faixas desgastadas, veículos e pedestres).

---

## 3. Exercício 3: Detecção de Formas Geométricas e Keypoints

### 3.1 Item A: Detecção de Sinais Circulares (`HoughCircles`) e Via (`HoughLinesP`) (`tp2-3a.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-3a.py
- **Comando para executar:** `python tp2-3a.py`

#### Resultados nas 5 Cenas de Tráfego (`rua_01.jpg` a `rua_05.jpg`) com 3 Configurações

| Imagem | Config 1 (`dp=1.0, p1=90, p2=23`) | Config 2 (`dp=1.2, p1=135, p2=29`) | Config 3 (`dp=1.5, p1=165, p2=40`) | Linhas `HoughLinesP` |
| :--- | :--- | :--- | :--- | :---: |
| **`rua_01.jpg`** | `3` círculos (`18.01 ms`) | **`1` placa**: `(820,205) r=40 c=1.00` (`7.81 ms`) | `1` placa: `(820,205) r=40 c=1.00` (`6.77 ms`) | `15` |
| **`rua_02.jpg`** | `4` círculos (`14.82 ms`) | **`3` candidatos**: `(825,230) r=19 c=1.00`; `(175,213) r=33 c=1.00` (`8.84 ms`) | `2` placas: `(175,214) r=33 c=1.00`; `(826,230) r=19 c=0.95` (`7.28 ms`) | `2` |
| **`rua_03.jpg`** | `3` círculos (`14.60 ms`) | **`2` candidatos**: `(115,165) r=48 c=1.00` (`10.41 ms`) | `1` placa: `(115,164) r=48 c=0.98` (`6.83 ms`) | `20` |
| **`rua_04.jpg`** | `2` círculos (`54.26 ms`) | **`3` candidatos**: `(775,227) r=13 c=0.94`; `(137,220) r=23 c=1.00` (`20.42 ms`) | `1` placa: `(137,220) r=23 c=1.00` (`12.15 ms`) | `36` |
| **`rua_05.jpg`** | `2` placas (`11.33 ms`) | **`2` placas**: `(785,190) r=28 c=1.00`; `(135,218) r=21 c=0.95` (`8.57 ms`) | `2` placas: `(785,190) r=28 c=1.00`; `(136,218) r=15 c=0.96` (`6.44 ms`) | `16` |

---

### 3.2 Item B: Harris vs Shi-Tomasi ($N=200$) e Fluxo Óptico Lucas-Kanade (`tp2-3b.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-3b.py
- **Comando para executar:** `python tp2-3b.py`

#### Comparação de Desempenho Harris vs Shi-Tomasi ($N=200$ Keypoints)

| Cena | Harris Keypoints | Tempo Harris (`ms`) | Shi-Tomasi Keypoints | Tempo Shi-Tomasi (`ms`) | Razão Tempo (H/S) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`rua_01.jpg`** | `198` | `46.540` | `200` | `6.481` | `7.18x` |
| **`rua_02.jpg`** | `200` | `43.040` | `200` | `6.253` | `6.88x` |
| **`rua_03.jpg`** | `200` | `47.289` | `200` | `6.560` | `7.21x` |
| **`rua_04.jpg`** | `200` | `57.904` | `200` | `6.169` | `9.39x` |
| **`rua_05.jpg`** | `96` | `16.830` | `199` | `6.211` | `2.71x` |
| **Média Global** | **`178.8`** | **`42.321 ms`** | **`199.8`** | **`6.335 ms`** | **`6.68x`** |

#### Rastreamento por Fluxo Óptico Piramidal Lucas-Kanade (`frame_00.png` $\to$ `frame_01.png` e Vídeo)
- **Keypoints Shi-Tomasi rastreados com sucesso:** **`200 / 200` (`100.0%`)** em **`7.986 ms`** (`winSize=21x21, maxLevel=3`).
- **Deslocamento mediano:** $dx = +1.156\text{ px}$, $dy = +0.343\text{ px}$, magnitude $|\mathbf{v}| = \mathbf{1.207\text{ px/frame}}$ (máximo = `9.217 px/frame`).
- **Velocidade óptica aparente:** **`14.48 px/s`** (a `12 FPS`).

#### Estimativa de Velocidade Relativa de Obstáculos via Fluxo Óptico
Pela equação de projeção perspectiva $x = f_x \frac{X}{Z} + c_x$, a derivada temporal do ponto projetado fornece o fluxo óptico:
$$
u = \dot{x} = \frac{f_x}{Z} V_x - \frac{x - c_x}{Z} V_z, \qquad v = \dot{y} = \frac{f_y}{Z} V_y - \frac{y - c_y}{Z} V_z
$$
1. **Movimento Transversal ($V_x, V_y$):** Conhecendo a distância $Z$ do obstáculo (via par estéreo, LiDAR ou homografia do plano da pista) e descontando o ego-movimento rotacional/translacional do veículo (via IMU e odometria de rodas), a velocidade transversal real em $\text{m/s}$ é recuperada por $V_x = \frac{u \cdot Z}{f_x}$.
2. **Velocidade Longitudinal e Time-to-Collision ($\text{TTC}$):** Quando um obstáculo se aproxima ($V_z < 0$), seus vetores de fluxo divergem radialmente a partir do Foco de Expansão (FOE). A razão entre a distância de imagem ao FOE e a magnitude divergente fornece diretamente o **Time-to-Collision** $\text{TTC} = \frac{Z}{|V_z|} \approx \frac{r}{\dot{r}}$ em segundos, mesmo sem calibração métrica absoluta de profundidade.

---

## 4. Exercício 4: Detecção de Objetos em Ambiente Simulado com MetaDrive

### 4.1 Item A: Detector de Faixas em Tempo Real no MetaDrive (200 Steps) (`tp2-4a.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-4a.py
- **Comando para executar:** `python tp2-4a.py`

#### Resumo da Execução em 200 Steps (`MetaDriveEnv` + Câmera Frontal `640x360`)
- **Steps executados:** `200` (ao longo de `3` episódios com política de condução aleatória).
- **Detecção de faixas:** `151 / 200` steps (`75.5%`) com pelo menos uma borda polinomial ajustada e **`102 / 200` steps (`51.0%`)** com ambas as bordas visíveis simultaneamente (nos demais steps a política aleatória leva o veículo às margens ou curvas fechadas).
- **Telemetria registrada em `tp2-4a_metricas_200_steps.csv`:**
  - **Velocidade média do agente:** **`7.06 km/h`** (máx = `14.65 km/h`)
  - **Desvio lateral médio:** **`-10.78 px`** (`-0.107 m` estimados, $\sigma = 50.27\text{ px}$)
  - **Raio de curvatura mediano:** **`1243.9 px`**
  - **Tempo médio do detector de faixas:** **`8.910 ms/frame`** (`112.2 FPS` isolado)

#### Comparação Real (`KITTI` e `Rodovia`) vs Simulado (`MetaDrive`)

| Domínio Avaliado | Segmentos Hough | Faixas Ajustadas | Desvio Lateral | Raio de Curvatura | Tempo (`ms`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Real Urbano (`frontal_kitti.png`)** | `20` | `2` (falsos no piso) | `-41.7 px` | `266 px` | `14.48` |
| **Real Rodovia (`pista_udacity.png`)** | `9` | `2` (faixas reais) | `-19.0 px` | `4074 px` | `9.55` |
| **Simulado (`MetaDrive Step 1`)** | `4` | `2` (faixas sintéticas) | `-13.6 px` | `845 px` | `8.91` |

---

### 4.2 Item B: Integração YOLOv8n no Loop do Simulador MetaDrive (500 Steps) (`tp2-4b.py`)

- **Script:** https://github.com/LcsGondra/VeiculosAutonomos/blob/main/tp2/tp2-4b.py
- **Comando para executar:** `python tp2-4b.py`

#### Degradação de FPS (Sem YOLO vs Com YOLOv8n Ativo)

| Configuração do Pipeline | Latência Média (`ms/step`) | Tempo Inferência YOLO (`ms`) | FPS Efetivo | Impacto Relativo |
| :--- | :---: | :---: | :---: | :---: |
| **Pipeline SEM YOLO (Baseline)** | `26.27 ms` | `0.00 ms` | **`38.07 FPS`** | Referência |
| **Pipeline COM YOLOv8n (`640px, skip=1`)** | `44.42 ms` | `16.55 ms` | **`22.51 FPS`** | **`-40.87%` de FPS** (`+18.15 ms`) |
| **Otimização 1: Frame Skipping 2x (`skip=2`)** | `34.20 ms` | `16.55 ms` (a cada 2 steps) | **`29.24 FPS`** | **`+29.9%` vs YOLO Base** |
| **Otimização 2: Frame Skipping 4x (`skip=4`)** | `32.04 ms` | `16.55 ms` (a cada 4 steps) | **`31.21 FPS`** | **`+38.7%` vs YOLO Base** |
| **Otimização 3: Resolução `320x320` (`skip=1`)** | `47.65 ms` | `17.80 ms` (overhead CPU resize) | **`20.99 FPS`** | `-6.8%` (melhor em GPU/NPU dedicada) |

#### Contagem de Objetos Detectados por Categoria em 500 Steps (`tp2-4b_yolo_500_steps.csv`)
- **Frames com detecções ativas (`>0` objetos):** **`469 / 500` frames (`93.8%`)**.
- **Totais acumulados por classe COCO:**
  - **`truck`:** **`1.243`** detecções (`2.486` objetos/frame; pico = `7`) — nota: devido ao *domain shift* das texturas geométricas low-poly do Panda3D/MetaDrive, veículos utilitários e sedãs angulares distantes são frequentemente classificados como `truck`.
  - **`car`:** **`414`** detecções (`0.828` objetos/frame; pico = `7`).
  - **`person`:** **`4`** detecções (`0.008` objetos/frame; pico = `1`).
  - **`bus`:** **`2`** detecções (`0.004` objetos/frame; pico = `1`).

#### Discussão Técnica: Impacto no Loop de Controle e Estratégias de Otimização
1. **Atraso de Fase no Loop de Controle:** A inserção síncrona do YOLOv8n elevou a latência do ciclo de `26.27 ms` (`38.07 FPS`) para `44.42 ms` (`22.51 FPS`), uma **degradação de `40.87%` no FPS**. Em um veículo real a $100\text{ km/h}$ ($27.78\text{ m/s}$), um acréscimo de $18.15\text{ ms}$ a $50\text{ ms}$ por frame adiciona $0.50\text{ m}$ a $1.39\text{ m}$ de atraso espacial na malha fechada de controle lateral (PID/MPC) e frenagem autônoma de emergência (AEB), reduzindo a margem de fase do controlador.
2. **Frame Skipping (`skip=2` e `skip=4`):** Executar o detector pesado a cada $k$ quadros e propagar as caixas nos quadros intermediários (via rastreador leve como Lucas-Kanade ou Filtro de Kalman/SORT) elevou o FPS de `22.51 FPS` para **`29.24 FPS` (`+29.9%`)** com `skip=2` e **`31.21 FPS` (`+38.7%`)** com `skip=4`, recuperando quase a taxa original do simulador.
3. **Redução de Resolução (`320x320` vs `640x640`) e Quantização:** Reduzir a entrada para `320x320` diminui em $4\times$ o volume espacial das ativações convolucionais (muito efetivo em GPUs/NPUs embarcadas como NVIDIA Jetson Orin com TensorRT INT8/FP16), porém compromete a detecção de objetos pequenos a longa distância (placas e pedestres distantes). Em arquiteturas de produção (ROS 2 / Autoware), desacopla-se a thread de percepção profunda (`20–30 Hz`) da thread de atuação/controle (`100 Hz`).


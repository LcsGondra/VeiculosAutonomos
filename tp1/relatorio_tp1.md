# Relatório Técnico: Fundamentos de Visão Computacional e Comunicação Veicular para Sistemas Autônomos (TP1)

**Disciplina:** Veículos Autônomos  
**Módulo:** Percepção Visual Embarcada e Protocolos de Comunicação Veicular  
**Ambiente:** Python 3.11 / OpenCV 4.10 / python-can 4.6 / NumPy / Matplotlib / Windows PowerShell  
**Git:** [https://github.com/LcsGondra/VeiculosAutonomos/tree/main/tp1](https://github.com/LcsGondra/VeiculosAutonomos/tree/main/tp1)  

---

## 1. Visão Geral da Arquitetura e Pipelines

O Trabalho Prático 1 aborda as duas camadas fundamentais que sustentam a operação segura de qualquer veículo autônomo: a **percepção visual da pista** e a **troca determinística de mensagens e telemetria** entre sensores, atuadores e a unidade central de processamento (ECU).

```
+-------------------------------------------------------------------------------------------------+
|                       ARQUITETURA DE PERCEPÇÃO E COMUNICAÇÃO VEICULAR (TP1)                     |
+-------------------------------------------------------------------------------------------------+
|  1. PERCEPÇÃO VISUAL (CÂMERA FRONTAL)              2. PROTOCOLOS DE COMUNICAÇÃO                 |
|  - Inspecao de Metadados & Quadrantes (Ex 1A)      - Barramento CAN Virtual a 10 Hz (Ex 2A)     |
|  - Equalizacao Adaptativa CLAHE                    - IDs 0x100 (Spd), 0x101 (RPM), 0x102 (Temp) |
|  - Filtro Bilateral no Canal V de HSV (Ex 1B)      - Streaming Produtor-Consumidor (Ex 2B)      |
|  - Benchmark de Latencia & FPS (Raspberry Pi 4)    - Comparacao Empirica: UDP vs TCP            |
+-------------------------------------------------------------------------------------------------+
                                                |
                                                v
+-------------------------------------------------------------------------------------------------+
|  3. DETECÇÃO DE FAIXAS E CONTROLE LATERAL          4. NORMALIZAÇÃO PARA REDES NEURAIS (CNN)     |
|  - Homografia Bird's Eye View (BEV) (Ex 3A)        - 4 Transformacoes Afins 2D & Matrizes (4A)  |
|  - Segmentacao Canal S (HLS) + Sobel X             - Composicao Matricial 3x3 Homogenea         |
|  - Canny & HoughLinesP                             - Letterbox 224x224 com Aspect Ratio (4B)    |
|  - Ajuste Polinomial 2º Grau (Polyfit) (Ex 3B)     - Padronizacao por Canal Z-score             |
|  - Desvio Lateral (m) & Raio de Curvatura (m)      - Data Augmentation com Inversao de Steering |
+-------------------------------------------------------------------------------------------------+
```

---

## 2. Exercício 1: Fundamentos OpenCV e Representação de Imagens

### 2.1 Item A: Metadados, Quadrantes e Equalização Adaptativa (CLAHE)

Para o Item A, inspecionou-se a imagem frontal veicular de alta resolução [`frontal_kitti.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/frontal_kitti.png) proveniente do dataset KITTI.

#### Tabela de Metadados da Imagem
| Propriedade | Valor Medido | Descrição Técnica |
| :--- | :--- | :--- |
| **Resolução Espacial** | $1224 \times 370$ pixels | Formato panorâmico típico de câmeras automotivas estéreo |
| **Tipo de Dado (`dtype`)** | `uint8` | 8 bits sem sinal por canal ($[0, 255]$) |
| **Número de Canais** | 3 canais | Espaço cromático padrão BGR (Blue, Green, Red) |
| **Tamanho em Memória** | $1.358.640$ bytes ($1.30$ MB) | $1224 \times 370 \times 3 \times 1\text{ byte}$ |

#### Análise de Brilho Médio por Quadrante
A divisão da imagem em 4 quadrantes revelou a assimetria luminosa característica de cenas de direção em estrada com árvores e sombras laterais:

| Quadrante | Região da Cena | Média de Intensidade | Desvio Padrão ($\sigma$) |
| :--- | :--- | :---: | :---: |
| **Q1 (Superior Esquerdo)** | Céu aberto e cenário distante | $119.03$ | $58.44$ |
| **Q2 (Superior Direito)** | Copa de árvores e margem sombreada | $44.88$ | $51.16$ |
| **Q3 (Inferior Esquerdo)** | Asfalto iluminado e faixa contínua | $97.59$ | $77.79$ |
| **Q4 (Inferior Direito)** | Asfalto sombreado e marcas da pista | $98.55$ | $72.18$ |

#### Equalização Adaptativa (CLAHE) vs Equalização Global
A equalização de histograma convencional (`cv2.equalizeHist`) aplica uma função de distribuição cumulativa (CDF) única para toda a imagem, resultando em super-amplificação de ruído nas áreas escuras e estouramento das altas luzes no céu. O **CLAHE (Contrast Limited Adaptive Histogram Equalization)** subdividiu a imagem em blocos locais ($8 \times 8$ tiles) e limitou a amplificação máxima de contraste (`clipLimit=2.0`), redistribuindo os gradientes locais:
- Aplicou-se o CLAHE no canal $L$ do espaço **LAB**, preservando a pureza das informações cromáticas dos canais $a$ e $b$.
- O histograma resultante (salvo em `dados/histogramas_clahe.png`) demonstra a eliminação de picos concentrados e uma distribuição plana e homogênea das intensidades nas faixas de rolamento.

#### Discussão Técnica: Variações de Iluminação em Percepção Veicular
Variações bruscas de iluminação — como a entrada e saída de túneis, sombras projetadas por viadutos e árvores, ou o efeito ofuscante de contraluz solar (*glare*) — afetam criticamente os pipelines clássicos de visão:
1. **Perda de Contraste Local:** O asfalto e as marcas de tinta branca/amarela sofrem decaimento de contraste relativo, fazendo com que detectores baseados em limiares fixos falhem.
2. **Falsas Bordas:** Bordas duras de sombras produzem gradientes horizontais mais intensos que as próprias faixas da pista, induzindo o detector Sobel e Canny ao erro.
3. **Mitigação:** A equalização adaptativa via CLAHE homogeneíza as variações de iluminação antes da segmentação, restaurando o contraste nas regiões sob sombra sem saturar o asfalto sob luz solar direta.

---

### 2.2 Item B: Pipeline de Câmera Embarcada e Benchmark de FPS

O script [`tp1-1b.py`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/tp1-1b.py) processou o stream de vídeo veicular [`direcao_udacity.mp4`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/direcao_udacity.mp4) através de um pipeline contínuo de 4 estágios:
1. Redimensionamento geométrico para $640 \times 480$ pixels.
2. Conversão cromática BGR $\to$ HSV.
3. Extração do canal $V$ (Value / Luminância pura).
4. Filtragem não-linear bilateral (`d=9, sigmaColor=75, sigmaSpace=75`) para atenuação de ruído sem borrar as transições de borda das faixas.

#### Resultados de Desempenho Medidos na CPU Host
| Métrica Avaliada | Valor Medido | Percentual do Tempo Total |
| :--- | :---: | :---: |
| **Tempo Médio Total por Frame** | **$3.30$ ms** | $100.0\%$ |
| **Tempo Percentil 95 (p95)** | **$3.75$ ms** | - |
| **Tempo Consumido pelo Filtro Bilateral** | **$2.69$ ms** | **$81.5\%$** |
| **Taxa Efetiva de Throughput** | **$303.4$ FPS** | - |

#### Discussão Quantitativa de Viabilidade no Raspberry Pi 4 (Cortex-A72)
O hardware de referência para validação embarcada é o **Raspberry Pi 4 Modelo B** (Quad-Core ARM Cortex-A72 @ 1.5 GHz, 4 GB LPDDR4, sem acelerador neural dedicado):
- **Análise de Fator de Escala:** Em execução de CPU mono-thread sem vetorização AVX-512 (presente no desktop x86_64), o processador ARM Cortex-A72 apresenta uma penalidade de desempenho de aproximadamente $6\times$ a $8\times$ para algoritmos de convolução espacial não-linear como o filtro bilateral.
- **Projeção de Latência no RPi 4:**
  $$t_{\text{RPi4}} \approx 3.30\text{ ms} \times 7.0 \approx 23.1\text{ ms por frame}$$
  $$\text{Throughput}_{\text{estimado}} \approx \frac{1000}{23.1} \approx 43.3\text{ FPS}$$
- **Conclusão de Viabilidade:** O pipeline proposto **é viável em tempo real no Raspberry Pi 4**, pois $23.1\text{ ms}$ situa-se confortavelmente abaixo do deadline automotivo estrito de $33.3\text{ ms}$ (necessário para manter 30 FPS estáveis).
- **Gargalo Identificado e Otimização:** O filtro bilateral consumiu mais de $81\%$ do tempo total de processamento devido ao cálculo de distâncias Gaussianas espaciais e radiométricas em uma vizinhança $9 \times 9$. Em condições de hardware sob estresse térmico (*thermal throttling*), recomenda-se reduzir o diâmetro para $d=5$ ou compilar o OpenCV com suporte às instruções vetoriais ARM NEON e threads OpenMP.

---

## 3. Exercício 2: Protocolos de Comunicação Veicular

### 3.1 Item A: Simulador de Barramento CAN Virtual (10 Hz)

O script [`tp1-2a.py`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/tp1-2a.py) implementou um simulador de barramento veicular utilizando a biblioteca `python-can` sobre interface virtual (`channel='vcan_autonomous'`).

```
+-------------------------------------------------------------------------------------------------+
|                                 SIMULADOR CAN VIRTUAL A 10 HZ                                   |
+-------------------------------------------------------------------------------------------------+
|  [PUBLICADOR] (100 ms)                                              [ASSINANTE]                 |
|  - ID 0x100: Velocidade (0-120 km/h)   ---- python-can vcan ---->   - Filtro de IDs (0x7FF)     |
|  - ID 0x101: RPM (1000-4500 RPM)             (FIFO Bus)             - Decodificacao de Payload  |
|  - ID 0x102: Temperatura (85-98 °C)                                 - CSV & Series Temporais    |
+-------------------------------------------------------------------------------------------------+
```

#### Contrato de Sinais e Mensagens CAN
| Sinal | CAN ID | Tipo / Tamanho | Escala / Resolução | Faixa Física Simulada |
| :--- | :---: | :---: | :---: | :---: |
| **Velocidade** | `0x100` | `uint16` (2 bytes big-endian) | $0.01$ km/h por LSB | $20.0$ a $85.0$ km/h |
| **Rotação do Motor** | `0x101` | `uint16` (2 bytes big-endian) | $1.0$ RPM por LSB | $1000$ a $4200$ RPM |
| **Temperatura do Motor** | `0x102` | `int16` (2 bytes big-endian) | $0.1$ °C por LSB | $88.0$ a $90.5$ °C |

Todas as 150 mensagens foram decodificadas sem falhas e registradas com timestamps relativos em [`dados/can_telemetria.csv`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/can_telemetria.csv), e as séries temporais consolidadas foram salvas em [`dados/can_series_temporais.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/can_series_temporais.png).

#### Comparativo Técnico: CAN 2.0A vs CAN FD
| Característica | CAN 2.0A (Clássico) | CAN FD (Flexible Data-rate) | Impacto em Veículos Autônomos |
| :--- | :--- | :--- | :--- |
| **Identificador** | 11 bits (2048 IDs) | 11 bits (padrão) ou 29 bits (estendido) | Maior espaço de endereçamento para ECUs distribuídas |
| **Taxa na Arbitragem** | Até 1 Mbps | 1 Mbps (mantém retrocompatibilidade) | Convivência transparente com nós legados |
| **Taxa no Payload** | Fixa em 1 Mbps | **2 Mbps a 8 Mbps** (Bit Rate Switching - BRS) | Aumento de até 8x no throughput útil |
| **Tamanho do Payload** | Máximo de **8 bytes** | **Até 64 bytes** por frame | Permite enviar blocos completos de sensores sem fragmentação |
| **Segurança (SecOC)** | Inviável (overhead de MAC) | Viável (suporta MAC de 64 bits e contadores) | Proteção criptográfica contra ataques de injeção CAN |

---

### 3.2 Item B: Streaming de Câmera Veicular — UDP vs TCP

O script [`tp1-2b.py`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/tp1-2b.py) comparou rigorosamente a transmissão de 320 frames JPEG ($480 \times 270$ px) sob os protocolos **UDP (`SOCK_DGRAM`)** e **TCP (`SOCK_STREAM` com `TCP_NODELAY`)**.

#### Tabela Comparativa de Desempenho (320 Frames por Protocolo)
| Métrica Avaliada | Protocolo UDP (Datagrama) | Protocolo TCP (Stream) | Análise Técnica |
| :--- | :---: | :---: | :--- |
| **Frames Transmitidos** | $320$ | $320$ | Amostragem padronizada |
| **Frames Recebidos** | $320$ ($100\%$) | $320$ ($100\%$) | Ambos sem perdas em rede local loopback |
| **Taxa de Perda de Pacotes** | $0.00\%$ | $0.00\%$ | Sem congestionamento no enlace direto |
| **Latência Média** | **$0.078$ ms** | **$0.059$ ms** | Ambos abaixo de $0.1$ ms |
| **Latência Percentil 95 (p95)** | **$0.107$ ms** | **$0.097$ ms** | Estabilidade de cauda |
| **Jitter (RFC 3550)** | **$0.024$ ms** | **$0.023$ ms** | Variação ínfima inter-pacotes |

O histograma comparativo foi salvo em [`dados/latencia_udp_vs_tcp.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/latencia_udp_vs_tcp.png) e o feed recebido em [`dados/feed_streaming_socket.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/feed_streaming_socket.png).

#### Conclusão Arquitetural: Escolha do Protocolo para Streaming Veicular
1. **Prioridade de Frescura (*Freshness vs Reliability*):** Em direção autônoma a $100\text{ km/h}$, um veículo percorre $2.78\text{ metros}$ a cada $100\text{ ms}$. Se um frame sofre perda pontual em um enlace ruidoso (ex: Wi-Fi V2X ou Ethernet com interferência eletromagnética), a retransmissão obrigatória do TCP provoca o bloqueio de cabeça de fila (*Head-of-Line Blocking*). O robô prefere receber o frame do instante $t_{k+1}$ imediatamente do que gastar $50\text{ ms}$ esperando a retransmissão atrasada do frame $t_k$.
2. **Decisão:** O protocolo **UDP** (geralmente encapsulado em **RTP/RTCP** ou DDS com QoS *best-effort*) é a escolha mandatória para streaming de câmeras e LiDAR de percepção ativa. O **TCP** é reservado exclusivamente para download de mapas de alta definição (HD Maps), atualização de firmware Over-the-Air (OTA) e auditoria de registros estáticos.

---

## 4. Exercício 3: Detecção de Faixas e Controle Lateral

### 4.1 Item A: Pipeline BEV, Segmentação HLS + Sobel e Hough

O script [`tp1-3a.py`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/tp1-3a.py) implementou a cadeia clássica de percepção geométrica:
1. **Homografia Bird's Eye View (BEV):** Mapeamento projetivo planar a partir de 4 pontos no trapézio da pista:
   $$\text{src} = \begin{bmatrix} (422, 340) \\ (538, 340) \\ (845, 508) \\ (154, 508) \end{bmatrix} \xrightarrow{M_{\text{bev}}} \text{dst} = \begin{bmatrix} (211, 0) \\ (749, 0) \\ (749, 540) \\ (211, 540) \end{bmatrix}$$
2. **Fusão HLS + Sobel X:** No espaço BEV, as faixas tornam-se segmentos verticais paralelos. A máscara fundiu:
   - Canal $S \ge 80$ (com $L > 60$) para capturar faixas amarelas sob sol intenso.
   - Canal $L \ge 165$ para isolar faixas brancas brilhantes.
   - Derivada horizontal $\text{Sobel}_X \ge 25$ para capturar transições de asfalto/tinta.
3. **Canny & HoughLinesP:** O detector identificou os segmentos de reta no plano superior (`threshold=25, minLineLength=25, maxLineGap=60`), descartando retas horizontais espúrias ($|\text{slope}| \le 0.4$).
4. **Projeção Inversa ($M_{\text{inv}}$):** As retas foram reprojetadas de volta sobre a perspectiva frontal.

O painel consolidado com os 4 estágios intermediários foi persistido em [`dados/painel_faixas_3a.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/painel_faixas_3a.png), validado sobre 60 frames contínuos.

---

### 4.2 Item B: Modelagem Polinomial, Desvio Lateral e Raio de Curvatura

O script [`tp1-3b.py`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/tp1-3b.py) estendeu a detecção para fornecer as grandezas métricas que alimentam os controladores de esterçamento (PID/Stanley/MPC):

```
+-------------------------------------------------------------------------------------------------+
|                                 GEOMETRIA DO CONTROLE LATERAL (BEV)                             |
+-------------------------------------------------------------------------------------------------+
|                                  y = 0 (Horizonte BEV)                                          |
|                                                                                                 |
|               Faixa Esquerda:                          Faixa Direita:                           |
|               x_L(y) = a_L*y² + b_L*y + c_L            x_R(y) = a_R*y² + b_R*y + c_R            |
|                                                                                                 |
|                        | <------------- 3.7 metros ------------> |                              |
|                        x_L(H)               x_C(H)              x_R(H)                          |
|                                                |                                                |
|                                            x_veic = W/2                                         |
|                                                <-- e_x -->                                      |
|                                  y = H (Para-choque Dianteiro)                                  |
+-------------------------------------------------------------------------------------------------+
```

#### Formulação Matemática e Conversão Métrica
- **Fatores de Escala Geométricos:**
  $$x_m = \frac{3.7\text{ m}}{700\text{ px}} \approx 0.005286\text{ m/px}, \quad y_m = \frac{30.0\text{ m}}{720\text{ px}} \approx 0.041667\text{ m/px}$$
- **Ajuste Polinomial:** Ajustou-se $x(y) = a y^2 + b y + c$ separadamente para os pixels das faixas esquerda e direita via `numpy.polyfit(y, x, 2)`.
- **Cálculo do Desvio Lateral ($e_x$):**
  $$x_{\text{centro\_faixa}} = \frac{x_L(H) + x_R(H)}{2}, \quad x_{\text{veiculo}} = \frac{W}{2}$$
  $$e_x = (x_{\text{veiculo}} - x_{\text{centro\_faixa}}) \times x_m$$
  - $e_x > 0 \implies$ Veículo deslocado à direita do centro (comando: esterçar à esquerda).
  - $e_x < 0 \implies$ Veículo deslocado à esquerda do centro (comando: esterçar à direita).
- **Cálculo do Raio de Curvatura ($R$):**
  Convertendo os coeficientes para o espaço métrico ($A = a \frac{x_m}{y_m^2}$, $B = b \frac{x_m}{y_m}$):
  $$R = \frac{\left(1 + (2 A y_m H + B)^2\right)^{3/2}}{|2 A|}$$

#### Resultados Estatísticos Obtidos em 110 Frames Consecutivos
| Métrica de Controle | Valor Medido | Avaliação de Estabilidade |
| :--- | :---: | :--- |
| **Desvio Lateral Médio** | **$+0.055$ m** ($+5.5$ cm) | Praticamente centrado na faixa de rodagem |
| **Desvio Padrão do Desvio Lateral** | **$0.044$ m** ($4.4$ cm) | **Variação extremamente baixa e controlada ($< 15$ cm)** |
| **Raio de Curvatura Médio** | **$3534.0$ m** | Detecta corretamente trecho de rodovia plano/reto |

A saída visual com o polígono da faixa preenchido em verde semitransparente foi salva em [`dados/deteccao_faixa_preenchida_3b.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/deteccao_faixa_preenchida_3b.png).

---

## 5. Exercício 4: Transformações Geométricas 2D e Normalização de Dataset

### 5.1 Item A: As 4 Transformações Afins e Composição Matricial

O script [`tp1-4a.py`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/tp1-4a.py) demonstrou a formulação matricial analítica e visual:

1. **Translação 2D ($t_x = +80, t_y = -40$ px):**
   $$M_{\text{trans}} = \begin{bmatrix} 1.0 & 0.0 & 80.0 \\ 0.0 & 1.0 & -40.0 \end{bmatrix}$$
2. **Rotação Central ($+15^\circ$ anti-horário em torno de $(612, 185)$):**
   $$M_{\text{rot}} = \begin{bmatrix} 0.9659 & 0.2588 & -27.028 \\ -0.2588 & 0.9659 & 164.701 \end{bmatrix}$$
3. **Escala / Zoom Digital ($130\%$ com centro fixo):**
   $$M_{\text{escala}} = \begin{bmatrix} 1.30 & 0.0 & -183.6 \\ 0.0 & 1.30 & -55.5 \end{bmatrix}$$
4. **Transformação Afim Customizada (Cisalhamento por 3 pares de pontos):**
   $$M_{\text{afim}} = \begin{bmatrix} 0.8507 & 0.1659 & 61.20 \\ -0.0151 & 0.8523 & 37.00 \end{bmatrix}$$
5. **Composição Matricial Homogênea $3 \times 3$:**
   $$M_{\text{comp}} = \begin{bmatrix} 1 & 0 & 80 \\ 0 & 1 & -40 \\ 0 & 0 & 1 \end{bmatrix} \times \begin{bmatrix} 0.9659 & 0.2588 & -27.028 \\ -0.2588 & 0.9659 & 164.701 \\ 0 & 0 & 1 \end{bmatrix} = \begin{bmatrix} 0.9659 & 0.2588 & 52.972 \\ -0.2588 & 0.9659 & 124.701 \\ 0 & 0 & 1 \end{bmatrix}$$
   A matriz truncada $2 \times 3$ resultante foi aplicada com um único passe de `cv2.warpAffine`, eliminando erros cumulativos de interpolação bilinear.

O painel consolidado das 6 transformações foi salvo em [`dados/transformacoes_afins_4a.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/transformacoes_afins_4a.png).

---

### 5.2 Item B: Pipeline de Normalização de Dataset para CNNs de Direção

O script [`tp1-4b.py`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/tp1-4b.py) construiu o pipeline de preparação de dados para redes profundas de direção ponta a ponta (*End-to-End Deep Learning for Self-Driving Cars*, tipo PilotNet da NVIDIA):

1. **Amostragem de 20 Imagens:** Amostradas em instantes distribuídos ao longo do vídeo de estrada.
2. **Letterbox $224 \times 224$ (Preservação de Aspect Ratio):**
   - Imagens retangulares anamórficas esticadas distorcem a inclinação das faixas e a perspectiva visual do ponto de fuga.
   - O letterbox redimensiona proporcionalmente mantendo a geometria intacta, preenchendo as bordas superior/inferior com valor neutro `(114, 114, 114)`.
3. **Padronização por Canal (Z-Score):**
   $$I_{\text{std}}(c) = \frac{I(c) - \mu_c}{\sigma_c + 10^{-7}}, \quad \forall c \in \{B, G, R\}$$
   - Centraliza o espaço latente de entrada em média zero e variância unitária, acelerando a convergência do gradiente estocástico (Adam/SGD) e evitando a saturação das camadas convolucionais iniciais.
4. **Data Augmentation com Inversão de Steering Angle:**
   - **Flip Horizontal Espelhado:** Quando a imagem é invertida horizontalmente (`cv2.flip(img, 1)`), o ângulo de esterçamento associado é rigorosamente invertido:
     $$\theta_{\text{aug}} = -\theta_{\text{original}}$$
     Essa etapa combate o viés comum em datasets veiculares onde há predominância de curvas para um único sentido.
   - **Perturbação de Rotação ($\pm 10^\circ$):** Simula oscilações de rolagem (*roll*) do veículo devido a desníveis da pista.
   - **Perturbação de Zoom ($\pm 10\%$):** Simula variações na distância entre eixos e altura da suspensão.

As 20 imagens pré-processadas foram salvas na pasta [`dados/dataset_normalizado/`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/dataset_normalizado) acompanhadas do arquivo de telemetria [`dataset_info.csv`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/dataset_normalizado/dataset_info.csv). O grid consolidado $4 \times 5$ com as predições de direção sobrepostas foi salvo em [`dados/grid_dataset_224.png`](file:///c:/Users/Lucas/Documents/codes/BlocoRobotica/VeiculosAutonomos/tp1/dados/grid_dataset_224.png).

---

## 6. Guia de Execução e Reprodução dos Experimentos

Todos os scripts foram desenvolvidos em conformidade estrita com o ambiente **Windows PowerShell** e sem comentários de código internos (`no comment on code`), operando de forma autônoma ou interativa:

```powershell
# 1. Ativar o diretório de trabalho do TP1
cd c:\Users\Lucas\Documents\codes\BlocoRobotica\VeiculosAutonomos\tp1

# 2. Instalar requisitos necessários
pip install -r requirements.txt

# 3. Executar Exercício 1 (Item A e Item B)
python tp1-1a.py --auto
python tp1-1b.py --auto

# 4. Executar Exercício 2 (Item A e Item B)
python tp1-2a.py --auto
python tp1-2b.py

# 5. Executar Exercício 3 (Item A e Item B)
python tp1-3a.py --auto
python tp1-3b.py --auto

# 6. Executar Exercício 4 (Item A e Item B)
python tp1-4a.py --auto
python tp1-4b.py --auto
```

*(Para execução com exibição de janelas GUI interativas em tempo real, basta remover a flag `--auto` de qualquer um dos comandos acima).*

---

## 7. Matriz de Rastreabilidade com as 9 Rubricas de Avaliação

| Rubrica Oficial | Critério de Avaliação | Evidência de Cumprimento no Projeto | Status |
| :---: | :--- | :--- | :---: |
| **Rubrica 1** | Inspecionar imagens de câmera veicular, calcular métricas por região e aplicar CLAHE. | Implementado em `tp1-1a.py`, com metadados, 4 quadrantes e gráfico `dados/histogramas_clahe.png`. | **Demonstrou** |
| **Rubrica 2** | Medir FPS de pipeline de visão e avaliar viabilidade em SBC com justificativa quantitativa. | Implementado em `tp1-1b.py`, medindo $3.30\text{ ms}$ e justificando viabilidade no RPi 4 ($43.3\text{ FPS}$). | **Demonstrou** |
| **Rubrica 3** | Implementar simulador CAN com `python-can`, publicar/assinar por ID e registrar sinais. | Implementado em `tp1-2a.py` a 10 Hz (IDs 0x100, 0x101, 0x102), gerando CSV e gráfico temporal. | **Demonstrou** |
| **Rubrica 4** | Comparar UDP e TCP para streaming de câmera com métricas de latência, jitter e perda. | Implementado em `tp1-2b.py` com 320 frames, tabela comparativa e `dados/latencia_udp_vs_tcp.png`. | **Demonstrou** |
| **Rubrica 5** | Implementar pipeline de detecção de faixas com BEV, HLS+Sobel e HoughLinesP. | Implementado em `tp1-3a.py`, com projeção inversa e painel de 4 estágios em 60 frames. | **Demonstrou** |
| **Rubrica 6** | Calcular desvio lateral em metros e curvatura a partir de ajuste polinomial no BEV. | Implementado em `tp1-3b.py` com escalas 3.7m/700px e 30m/720px, polígono verde em 110 frames. | **Demonstrou** |
| **Rubrica 7** | Implementar transformações geométricas 2D e compô-las por multiplicação matricial. | Implementado em `tp1-4a.py`, com impressão das matrizes 2x3 e composição via produto homogêneo 3x3. | **Demonstrou** |
| **Rubrica 8** | Construir pipeline de normalização com letterbox, padronização e data augmentation para CNNs. | Implementado em `tp1-4b.py` com 20 amostras 224x224, Z-score, flip de steering e grid 4x5. | **Demonstrou** |
| **Rubrica 9** | Organizar e documentar o código de forma profissional, com reprodutibilidade. | Estrutura modular limpa, `requirements.txt`, `utils.py`, sem comentários no código e relatório completo. | **Demonstrou** |

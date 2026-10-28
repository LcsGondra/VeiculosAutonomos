import os
import socket
import struct
import sys
import threading
import time
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from utils import calculate_rfc3550_jitter, imshow_keep_aspect_ratio

base_dir = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(base_dir, "dados", "direcao_udacity.mp4")
plot_path = os.path.join(base_dir, "dados", "latencia_udp_vs_tcp.png")

TOTAL_FRAMES = 320
TCP_PORT = 15001
UDP_PORT = 15002
HOST = "127.0.0.1"
HEADER_FORMAT = ">IId"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def get_video_frames(num_frames):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao carregar video {video_path}")
        sys.exit(1)

    frames = []
    while len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        resized = cv2.resize(frame, (480, 270))
        frames.append(resized)
    cap.release()
    return frames


def run_tcp_server(frames, ready_event):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((HOST, TCP_PORT))
    server_sock.listen(1)
    ready_event.set()

    conn, _ = server_sock.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    for i, frame in enumerate(frames):
        success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not success:
            continue
        data = enc.tobytes()
        t_send = time.perf_counter()
        header = struct.pack(HEADER_FORMAT, i + 1, len(data), t_send)
        try:
            conn.sendall(header + data)
        except (BrokenPipeError, ConnectionResetError):
            break
        time.sleep(0.015)

    conn.close()
    server_sock.close()


def run_tcp_client(num_frames, results_dict, auto_mode=True):
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    time.sleep(0.05)
    client_sock.connect((HOST, TCP_PORT))

    latencies = []
    received_count = 0
    last_frame = None
    win_name = "TP1 - Ex 2B: Cliente Streaming TCP"
    if not auto_mode:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 640, 360)

    def recv_exact(n):
        buf = bytearray()
        while len(buf) < n:
            chunk = client_sock.recv(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    while received_count < num_frames:
        header_bytes = recv_exact(HEADER_SIZE)
        if not header_bytes:
            break

        frame_id, length, t_send = struct.unpack(HEADER_FORMAT, header_bytes)
        payload = recv_exact(length)
        if not payload:
            break

        t_recv = time.perf_counter()
        arr = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        latency_ms = (t_recv - t_send) * 1000.0
        latencies.append(latency_ms)
        received_count += 1

        if decoded is not None:
            if received_count == num_frames // 2:
                last_frame = decoded.copy()
            if not auto_mode:
                hud = decoded.copy()
                cv2.putText(
                    hud,
                    f"TCP Frame {frame_id}/{num_frames} | Lat: {latency_ms:.2f} ms",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                imshow_keep_aspect_ratio(win_name, hud)
                cv2.waitKey(1)

    if not auto_mode:
        cv2.destroyWindow(win_name)
    client_sock.close()
    results_dict["latencies"] = latencies
    results_dict["received"] = received_count
    results_dict["lost"] = num_frames - received_count
    results_dict["sample"] = last_frame


def run_udp_server(frames, ready_event):
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ready_event.set()
    time.sleep(0.05)

    for i, frame in enumerate(frames):
        success, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        if not success:
            continue
        data = enc.tobytes()
        t_send = time.perf_counter()
        header = struct.pack(HEADER_FORMAT, i + 1, len(data), t_send)
        packet = header + data
        if len(packet) <= 65507:
            server_sock.sendto(packet, (HOST, UDP_PORT))
        time.sleep(0.015)

    server_sock.close()


def run_udp_client(num_frames, results_dict, auto_mode=True):
    client_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    client_sock.bind((HOST, UDP_PORT))
    client_sock.settimeout(1.5)

    latencies = []
    received_ids = set()
    last_frame = None
    win_name = "TP1 - Ex 2B: Cliente Streaming UDP"
    if not auto_mode:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 640, 360)

    start_wait = time.time()
    while len(received_ids) < num_frames and (time.time() - start_wait) < 8.0:
        try:
            packet, _ = client_sock.recvfrom(65535)
            t_recv = time.perf_counter()
        except socket.timeout:
            break

        if len(packet) < HEADER_SIZE:
            continue

        frame_id, length, t_send = struct.unpack(HEADER_FORMAT, packet[:HEADER_SIZE])
        payload = packet[HEADER_SIZE : HEADER_SIZE + length]
        arr = np.frombuffer(payload, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        latency_ms = (t_recv - t_send) * 1000.0
        latencies.append(latency_ms)
        received_ids.add(frame_id)

        if decoded is not None:
            if len(received_ids) == num_frames // 2:
                last_frame = decoded.copy()
            if not auto_mode:
                hud = decoded.copy()
                cv2.putText(
                    hud,
                    f"UDP Frame {frame_id}/{num_frames} | Lat: {latency_ms:.2f} ms",
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                imshow_keep_aspect_ratio(win_name, hud)
                cv2.waitKey(1)

    if not auto_mode:
        cv2.destroyWindow(win_name)
    client_sock.close()
    results_dict["latencies"] = latencies
    results_dict["received"] = len(received_ids)
    results_dict["lost"] = num_frames - len(received_ids)
    results_dict["sample"] = last_frame


def main():
    auto_mode = "--auto" in sys.argv
    print("=" * 65)
    print("SISTEMA PRODUTOR-CONSUMIDOR: STREAMING DE CAMERA VEICULAR")
    print(f"Protocolos Avaliados: UDP (SOCK_DGRAM) vs TCP (SOCK_STREAM)")
    print(f"Meta de Amostragem: {TOTAL_FRAMES} frames transmitidos por protocolo")
    print("=" * 65)

    frames = get_video_frames(TOTAL_FRAMES)

    print("Iniciando Transmissao via TCP (Controle de Fluxo e Confiabilidade)...")
    tcp_ready = threading.Event()
    tcp_results = {}
    t_tcp_srv = threading.Thread(target=run_tcp_server, args=(frames, tcp_ready))
    t_tcp_cli = threading.Thread(target=run_tcp_client, args=(TOTAL_FRAMES, tcp_results, auto_mode))
    t_tcp_srv.start()
    tcp_ready.wait()
    t_tcp_cli.start()
    t_tcp_srv.join()
    t_tcp_cli.join()
    print("Transmissao TCP Finalizada.")

    print("Iniciando Transmissao via UDP (Datagrama de Baixa Latencia)...")
    udp_ready = threading.Event()
    udp_results = {}
    t_udp_srv = threading.Thread(target=run_udp_server, args=(frames, udp_ready))
    t_udp_cli = threading.Thread(target=run_udp_client, args=(TOTAL_FRAMES, udp_results, auto_mode))
    t_udp_srv.start()
    udp_ready.wait()
    t_udp_cli.start()
    t_udp_srv.join()
    t_udp_cli.join()
    print("Transmissao UDP Finalizada.")

    tcp_lat = tcp_results.get("latencies", [0.0])
    udp_lat = udp_results.get("latencies", [0.0])

    tcp_mean = float(np.mean(tcp_lat))
    tcp_std = float(np.std(tcp_lat))
    tcp_p95 = float(np.percentile(tcp_lat, 95))
    tcp_jitter = calculate_rfc3550_jitter(tcp_lat)
    tcp_loss_pct = (tcp_results["lost"] / TOTAL_FRAMES) * 100.0

    udp_mean = float(np.mean(udp_lat))
    udp_std = float(np.std(udp_lat))
    udp_p95 = float(np.percentile(udp_lat, 95))
    udp_jitter = calculate_rfc3550_jitter(udp_lat)
    udp_loss_pct = (udp_results["lost"] / TOTAL_FRAMES) * 100.0

    print("=" * 65)
    print("TABELA CONSOLIDADA DE DESEMPENHO: UDP vs TCP")
    print(f"{'Metrica':<28} | {'UDP (Datagrama)':<16} | {'TCP (Stream)':<16}")
    print("-" * 65)
    print(f"{'Frames Transmitidos':<28} | {TOTAL_FRAMES:<16} | {TOTAL_FRAMES:<16}")
    print(f"{'Frames Recebidos':<28} | {udp_results['received']:<16} | {tcp_results['received']:<16}")
    print(f"{'Taxa de Perda (%)':<28} | {udp_loss_pct:<15.2f}% | {tcp_loss_pct:<15.2f}%")
    print(f"{'Latencia Media (ms)':<28} | {udp_mean:<16.3f} | {tcp_mean:<16.3f}")
    print(f"{'Latencia p95 (ms)':<28} | {udp_p95:<16.3f} | {tcp_p95:<16.3f}")
    print(f"{'Desvio Padrao Latencia (ms)':<28} | {udp_std:<16.3f} | {tcp_std:<16.3f}")
    print(f"{'Jitter RFC 3550 (ms)':<28} | {udp_jitter:<16.3f} | {tcp_jitter:<16.3f}")
    print("=" * 65)

    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, max(max(tcp_lat), max(udp_lat)) * 1.1, 40)
    ax.hist(udp_lat, bins=bins, alpha=0.65, label=f"UDP (Media: {udp_mean:.2f} ms | Jitter: {udp_jitter:.2f} ms)", color="#2ca02c")
    ax.hist(tcp_lat, bins=bins, alpha=0.55, label=f"TCP (Media: {tcp_mean:.2f} ms | Jitter: {tcp_jitter:.2f} ms)", color="#1f77b4")
    ax.axvline(udp_mean, color="#1b611b", linestyle="--", linewidth=1.5)
    ax.axvline(tcp_mean, color="#0f446b", linestyle="--", linewidth=1.5)
    ax.set_title("Histograma Comparativo de Latencia de Transmissao: UDP vs TCP (320 Frames)")
    ax.set_xlabel("Latencia Fim-a-Fim (ms)")
    ax.set_ylabel("Frequencia de Frames")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Histograma de latencia salvo em: {plot_path}")

    sample_img = tcp_results.get("sample")
    if sample_img is not None:
        cv2.putText(
            sample_img,
            f"TCP Lat: {tcp_mean:.2f}ms | UDP Lat: {udp_mean:.2f}ms",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        sample_path = os.path.join(base_dir, "dados", "feed_streaming_socket.png")
        cv2.imwrite(sample_path, sample_img)
        print(f"Amostra do feed recebido salva em: {sample_path}")

    print("=" * 65)
    print("CONCLUSAO ARQUITETURAL: PROTOCOLO PARA STREAMING VEICULAR")
    print("1. O UDP e o protocolo mais indicado para o pipeline de percepcao em tempo real,")
    print("   pois prioriza a frescura do dado (staleness) sobre a retransmissao de frames atrasados.")
    print("2. Em direcao autonoma a 100 km/h, um frame atrasado em 100 ms por retransmissao TCP")
    print("   representa 2.78 metros de deslocamento as cegas. O veiculo prefere descartar um frame corrompido")
    print("   e processar o quadro mais recente do que enfileirar buffers de retransmissao (Head-of-Line Blocking).")
    print("3. O TCP e reservado para telemetria estatica, envio de logs, download de mapas HD")
    print("   e calibracao de sensores, onde a integridade completa dos bytes e obrigatoria.")
    print("=" * 65)


if __name__ == "__main__":
    main()

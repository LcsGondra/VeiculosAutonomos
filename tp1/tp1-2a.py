import csv
import math
import os
import struct
import sys
import threading
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import can
import pandas as pd

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "dados", "can_telemetria.csv")
plot_path = os.path.join(base_dir, "dados", "can_series_temporais.png")

CAN_CHANNEL = "vcan_autonomous"
ID_SPEED = 0x100
ID_RPM = 0x101
ID_TEMP = 0x102

stop_event = threading.Event()
records = []


def encode_speed(kmh):
    val = int(round(kmh * 100))
    return struct.pack(">H", val)


def decode_speed(payload):
    (val,) = struct.unpack(">H", payload[:2])
    return val / 100.0


def encode_rpm(rpm):
    val = int(round(rpm))
    return struct.pack(">H", val)


def decode_rpm(payload):
    (val,) = struct.unpack(">H", payload[:2])
    return float(val)


def encode_temp(celsius):
    val = int(round(celsius * 10))
    return struct.pack(">h", val)


def decode_temp(payload):
    (val,) = struct.unpack(">h", payload[:2])
    return val / 10.0


def publisher_thread(total_cycles):
    bus = can.interface.Bus(channel=CAN_CHANNEL, interface="virtual")
    cycle = 0
    start_time = time.time()

    while not stop_event.is_set():
        now = time.time() - start_time

        sim_speed = 50.0 + 30.0 * math.sin(now * 0.5) + 5.0 * math.sin(now * 2.0)
        sim_rpm = 1000.0 + (sim_speed / 120.0) * 3500.0 + 50.0 * math.sin(now * 3.0)
        sim_temp = 88.0 + 4.0 * math.sin(now * 0.1)

        msg_speed = can.Message(
            arbitration_id=ID_SPEED,
            data=encode_speed(sim_speed),
            is_extended_id=False,
            timestamp=time.time(),
        )
        msg_rpm = can.Message(
            arbitration_id=ID_RPM,
            data=encode_rpm(sim_rpm),
            is_extended_id=False,
            timestamp=time.time(),
        )
        msg_temp = can.Message(
            arbitration_id=ID_TEMP,
            data=encode_temp(sim_temp),
            is_extended_id=False,
            timestamp=time.time(),
        )

        bus.send(msg_speed)
        bus.send(msg_rpm)
        bus.send(msg_temp)

        cycle += 1
        if total_cycles > 0 and cycle >= total_cycles:
            time.sleep(0.05)
            stop_event.set()
            break

        time.sleep(0.10)

    bus.shutdown()


def subscriber_thread():
    bus = can.interface.Bus(
        channel=CAN_CHANNEL,
        interface="virtual",
        can_filters=[
            {"can_id": ID_SPEED, "can_mask": 0x7FF},
            {"can_id": ID_RPM, "can_mask": 0x7FF},
            {"can_id": ID_TEMP, "can_mask": 0x7FF},
        ],
    )

    t_init = time.time()

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "relative_time_s",
                "can_id_hex",
                "signal_name",
                "raw_hex",
                "physical_value",
                "unit",
            ]
        )

        while not stop_event.is_set():
            msg = bus.recv(timeout=0.2)
            if msg is None:
                continue

            rel_t = msg.timestamp - t_init
            hex_data = msg.data.hex().upper()

            if msg.arbitration_id == ID_SPEED:
                sig_name = "Velocidade"
                val = decode_speed(msg.data)
                unit = "km/h"
            elif msg.arbitration_id == ID_RPM:
                sig_name = "Rotacao_Motor"
                val = decode_rpm(msg.data)
                unit = "RPM"
            elif msg.arbitration_id == ID_TEMP:
                sig_name = "Temperatura_Arrefecimento"
                val = decode_temp(msg.data)
                unit = "degC"
            else:
                continue

            writer.writerow(
                [
                    f"{msg.timestamp:.4f}",
                    f"{rel_t:.3f}",
                    f"0x{msg.arbitration_id:03X}",
                    sig_name,
                    hex_data,
                    f"{val:.2f}",
                    unit,
                ]
            )

            records.append(
                {
                    "t": rel_t,
                    "id": msg.arbitration_id,
                    "signal": sig_name,
                    "val": val,
                    "unit": unit,
                }
            )

            if len(records) <= 6 or len(records) % 14 == 0:
                print(
                    f"CAN Rx: t={rel_t:6.2f}s | ID=0x{msg.arbitration_id:03X} | "
                    f"{sig_name:26s} | Raw={hex_data:6s} | Valor={val:7.2f} {unit}"
                )

    bus.shutdown()


def plot_series():
    if not records:
        print("Aviso: nenhum dado registrado para plotar.")
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

    df_spd = df[df["id"] == ID_SPEED]
    axes[0].plot(df_spd["t"], df_spd["val"], color="#1f77b4", linewidth=1.8, marker="o", markersize=3)
    axes[0].set_ylabel("Velocidade (km/h)")
    axes[0].set_title("Telemetria do Barramento CAN Virtual (Frequencia de Publicacao: 10 Hz)")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    df_rpm = df[df["id"] == ID_RPM]
    axes[1].plot(df_rpm["t"], df_rpm["val"], color="#ff7f0e", linewidth=1.8, marker="s", markersize=3)
    axes[1].set_ylabel("Rotacao (RPM)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    df_tmp = df[df["id"] == ID_TEMP]
    axes[2].plot(df_tmp["t"], df_tmp["val"], color="#2ca02c", linewidth=1.8, marker="^", markersize=3)
    axes[2].set_ylabel("Temperatura (degC)")
    axes[2].set_xlabel("Tempo Relativo de Execucao (s)")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Grafico das series temporais salvo com sucesso em: {plot_path}")


def main():
    print("=" * 65)
    print("SIMULADOR DE BARRAMENTO CAN VIRTUAL (10 HZ - SENSORES VEICULARES)")
    print("=" * 65)
    print("Topologia e Parametros:")
    print(" - Interface: python-can Virtual Bus (channel='vcan_autonomous')")
    print(" - Frequencia de Envio: 10 Hz (intervalo estrito de 100 ms)")
    print(" - ID 0x100: Velocidade do Veiculo (2 bytes uint16, resolucao 0.01 km/h)")
    print(" - ID 0x101: Rotacao do Motor (2 bytes uint16, resolucao 1 RPM)")
    print(" - ID 0x102: Temperatura do Arrefecimento (2 bytes int16, resolucao 0.1 degC)")
    print("=" * 65)

    auto_mode = "--auto" in sys.argv
    cycles = 50 if auto_mode else 100

    sub_t = threading.Thread(target=subscriber_thread)
    pub_t = threading.Thread(target=publisher_thread, args=(cycles,))

    sub_t.start()
    time.sleep(0.05)
    pub_t.start()

    pub_t.join()
    time.sleep(0.2)
    stop_event.set()
    sub_t.join()

    print("=" * 65)
    print(f"Simulacao concluida. Total de mensagens decodificadas: {len(records)}")
    print(f"Arquivo CSV de telemetria persistido em: {csv_path}")

    plot_series()

    print("=" * 65)
    print("ANALISE COMPARATIVA: CAN 2.0A vs CAN FD")
    print("1. Formato do Identificador e Arbitracao:")
    print("   - CAN 2.0A: Identificador padrao de 11 bits (2048 IDs possiveis).")
    print("   - CAN FD: Suporta identificador padrao (11 bits) e estendido (29 bits),")
    print("     mantendo retrocompatibilidade no barramento fisico.")
    print("2. Taxa de Dados (Throughput):")
    print("   - CAN 2.0A: Taxa unica e fixa limitada a 1 Mbps em toda a mensagem.")
    print("   - CAN FD (Flexible Data-rate): Transmite a fase de arbitragem a 1 Mbps")
    print("     e acelera a fase de payload de dados para 2 Mbps a 8 Mbps.")
    print("3. Carga Util (Payload Maximo):")
    print("   - CAN 2.0A: Maximo de 8 bytes por frame (alto overhead para sensores complexos).")
    print("   - CAN FD: Ate 64 bytes de payload por frame (8x maior que o CAN 2.0A).")
    print("4. Conclusao para Arquitetura de Veiculos Autonomos:")
    print("   - O CAN FD reduz o congestionamento do barramento, viabiliza o envio de matrizes")
    print("     de covariancia de IMU/GNSS, assinaturas criptograficas (SecOC) e telemetria")
    print("     densa de atuadores de direcao/frenagem sem fragmentacao de pacotes.")
    print("=" * 65)


if __name__ == "__main__":
    main()

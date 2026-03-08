import math
import random
from dataclasses import dataclass
from typing import Optional

# Constantes físicas globais
BOLTZMANN_CONST = 1.380649e-23


@dataclass
class Physics5GConfig:
    """Configuração dos parâmetros físicos para cálculo de latência 5G.

    Centraliza constantes mágicas anteriormente espalhadas no código.
    """

    distancia_padrao_m: float = 750.0
    potencia_transmissao_dbm: float = 30.0
    largura_banda_hz: float = 100e6
    temperatura_kelvin: float = 290.0
    figura_ruido_db: float = 10.0
    eficiencia_codec: float = 0.5
    snr_minimo_db: float = 0.0
    freq_portadora_hz: float = 3.5e9
    sigma_shadowing_db: float = 0.0


class NetworkPhysics:
    """Motor de cálculo de física de redes (Latência Computacional e 5G)."""

    @staticmethod
    def calculate_5g_latency(
        packet_size_mb: float,
        distance_m: Optional[float] = None,
        config: Physics5GConfig = Physics5GConfig(),
    ) -> float:
        """Calcula a latência de transmissão 5G baseada em Shannon-Hartley e Path Loss.

        Args:
            packet_size_mb: Tamanho do pacote em Megabits (não MByte).
                            Nota: O código original usava 'dado' vindo de bandwidth.
            distance_m: Distância entre transmissor e receptor em metros.
                        Se None, usa o padrão da config.
            config: Objeto de configuração com parâmetros físicos.

        Returns:
            float: Latência estimada em milissegundos (ms).
        """
        dist = distance_m if distance_m is not None else config.distancia_padrao_m

        # Evita log de 0 ou distância negativa
        if dist <= 0:
            dist = 1.0

        # 1. Path Loss (Modelo Log-Distance + Shadowing)
        # PL(d) = PL(d0) + 10n log10(d/d0) + Xg (simplificado no código original)
        pl_db = 28.0 + 22 * math.log10(dist) + 20 * math.log10(config.freq_portadora_hz / 1e9)

        # Adiciona variação estocástica (Shadowing)
        if config.sigma_shadowing_db > 0:
            pl_db += random.gauss(0, config.sigma_shadowing_db)

        ganho_canal = 10 ** (-pl_db / 10)

        # 2. Cálculo de SNR Linear
        potencia_w = (10 ** (config.potencia_transmissao_dbm / 10)) / 1000.0
        ruido_w_hz = (
            BOLTZMANN_CONST * config.temperatura_kelvin * (10 ** (config.figura_ruido_db / 10))
        )

        snr_linear = (ganho_canal * potencia_w) / (ruido_w_hz * config.largura_banda_hz)
        snr_min_linear = 10 ** (config.snr_minimo_db / 10)
        snr_linear = max(snr_linear, snr_min_linear)

        # 3. Capacidade do Canal (Shannon) e Latência
        taxa_bps = config.largura_banda_hz * math.log2(1 + snr_linear) * config.eficiencia_codec

        if taxa_bps <= 0:
            return float("inf")

        # O 'dado' original parecia vir de bw_required (Mbps?) ou size?
        # Assumindo conversão direta para manter compatibilidade com v2
        latencia_ms = (packet_size_mb / taxa_bps) * 1000.0

        return latencia_ms

    @staticmethod
    def calculate_processing_latency(bw_required: float, node_ips: float) -> float:
        """Calcula a latência de processamento em um nó (Computational Delay).

        Args:
            bw_required: Demanda de banda/processamento da VNF (entrada).
            node_ips: Capacidade de Instruções Por Segundo do nó.

        Returns:
            float: Latência de processamento em ms.
        """
        if node_ips <= 0:
            return float("inf")

        # Lógica original: (outcome / 60 * 1e6) * 10 * 1000 / ips
        packet_load = (bw_required / 60.0) * 1e6
        latency_ms = (packet_load * 10.0 * 1000.0) / node_ips

        return latency_ms

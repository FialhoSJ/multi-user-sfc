from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from typing import List

# 1. Definição do ROOT_PATH usando Pathlib (Seguro e Multiplataforma)
# Isso pega a pasta 'src/muar_sfc' e sobe os níveis necessários até a raiz do repositório
ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

class SimulationSettings(BaseSettings):
    """Configurações unificadas e validadas do Simulador SFC."""

    # Permite ler variáveis de um arquivo .env na raiz (Segurança e Observabilidade)
    model_config = SettingsConfigDict(env_prefix="sfc_", env_file=".env", extra="ignore")

    # --- Parâmetros de Entrada Obrigatórios ---
    n_sessions: int

    # --- Parâmetros com Valores Padrão ---
    ia: float = 0.0
    number_of_nodes: int = 36
    src_node: int = 0
    max_duration: int = 120
    latency: float = 6.0
    fator: float = 0.25
    average_time_session_arrival: float = 10.0
    ia_bw: float = 150.0
    cpb: float = 10e6
    ca_size: float = 240.0
    chr: float = 1.0 / 3.0
    ft_bw: float = 0.144
    det_bw: float = 0.230

    # --- Campos Derivados (Serão calculados automaticamente) ---
    ma_bw: float = 0.0
    ma: float = 0.0
    uni_bw: float = 0.0
    uni: float = 0.0
    re_bw: float = 0.0
    re: float = 0.0
    ec_tc_bw: float = 0.0
    ec_tc: float = 0.0
    det: float = 0.0
    ia_det_ft_bw: float = 0.0
    ia_det_ft: float = 0.0
    total: float = 0.0
    mono: float = 0.0
    ft: float = 0.0

    @model_validator(mode="after")
    def calculate_derived_parameters(self) -> "SimulationSettings":
        """Calcula os parâmetros derivados. Isso substitui o antigo __post_init__ defeituoso."""
        self.ma_bw = int(self.ca_size * self.chr)
        self.ma = self.ma_bw * self.cpb
        self.uni_bw = int(self.ca_size * (1 - self.chr))
        self.uni = self.uni_bw * self.cpb
        self.re_bw = self.ma_bw + self.uni_bw
        self.re = int(self.re_bw * self.cpb)
        self.ec_tc_bw = int(self.ia_bw * 0.9 * 0.9 * 0.8)
        self.ec_tc = self.ec_tc_bw * self.cpb
        self.det = self.det_bw * self.cpb
        self.ia_det_ft_bw = int(self.ia_bw + self.det_bw + self.ft_bw)
        self.ia_det_ft = int(self.ia + self.det + self.ft)
        self.total = self.ia + self.det + self.ft + self.ma + self.uni + self.re + self.ec_tc
        self.mono = int(self.det + self.ft + self.ma + self.uni + self.re + self.ec_tc)
        self.ft = self.ft_bw * self.cpb

        return self




class SimulationSettings(BaseSettings):
    """
    Configurações centralizadas do simulador Muar-SFC.
    O Pydantic fará a conversão automática de tipos e validação.
    """
    # --- Parâmetros Gerais e de Log ---
    application: str = "muar"
    alg: str = "vegeta"
    sfc_lifetime: int = 120
    verbose: str = "y"  # Nota: no futuro, podemos refatorar para bool!

    # --- Parâmetros de Topologia e Rede ---
    topology: str = "luxembourgv2"
    eco_effi_ratio: float = 0.7
    mobility: str = "n"

    # --- Parâmetros de Tráfego (Sessões/Jogadores) ---
    n_sessions: int = 50
    n_players: int = 6

    # --- Parâmetros SFC ---
    allow_md_host: str = "y"
    sfc: str = "on"
    share: str = "y"
    shareband: str = "n"
    allow_delay: str = "n"

    # --- Parâmetros de Confiabilidade e Falhas ---
    backup: str = "n"
    ava: float = 0.99  # Pydantic converte automaticamente!
    number_of_fails: int = 3
    min_fail_duration: float = 20.0
    crash_at: List[float] = [180.0, 520.0, 640.0]

    # --- Configuração MICRO (Confiabilidade Base por Nível) ---
    rel_high: float = 0.999
    rel_normal: float = 0.98
    rel_low: float = 0.95

    # --- Fator de Estresse ---
    stress_high: float = 0.02
    stress_normal: float = 0.08
    stress_low: float = 0.15

    fail_target: str = "all"
    link_ava: float = 0.95
    number_of_link_fails: int = 0
    min_link_fail_duration: float = 20.0

    # Configuração do Pydantic (permite ler de arquivo .env e prefixos)
    model_config = SettingsConfigDict(env_prefix="MUAR_", env_file=".env", env_file_encoding="utf-8")

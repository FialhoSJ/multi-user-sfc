import copy
import random

from muar_sfc.core.infrastructure.enums import SHAREABLE_PREFIXES
from muar_sfc.core.vnf import VNFType


class ServiceTemplate:
    """
    Template de um tipo de serviço.

    Cada serviço define sua cadeia de VNFs (mesmo formato de dict usado pelo
    VNFGenerator/SFCGenerator), o SLA de latência total e regras de
    compartilhamento.
    """

    def __init__(
        self,
        name: str,
        latency_request: float,
        vnf_chain: list[dict],
        shareable: bool = False,
        shareable_prefixes: tuple[str, ...] = (),
    ):
        self.name = name
        self.latency_request = latency_request
        self.vnf_chain = vnf_chain
        self.shareable = shareable
        self.shareable_prefixes = shareable_prefixes or (name.upper() + "_",)

    def build_chain(self, player: int = 1, counter: int = 0) -> list[dict]:
        """
        Materializa a cadeia substituindo os placeholders {player}/{counter}
        no nome das VNFs (para nomes únicos por sessão).
        """
        chain = []
        for vnf in self.vnf_chain:
            item = copy.deepcopy(vnf)
            item["name"] = (
                item["name"]
                .replace("{player}", str(player))
                .replace("{counter}", str(counter))
            )
            item.setdefault("service_type", self.name)
            item["shareable"] = self.shareable  # F3: compartilhamento por serviço
            chain.append(item)
        return chain

    def build_sfc_dict(
        self,
        sfc_name: str,
        src_node,
        dst_node,
        closer_router,
        bandwidth,
        duration,
        player: int = 1,
        counter: int = 0,
    ) -> dict:
        """Monta o dict completo no formato que o SFCGenerator espera."""
        return {
            "name": sfc_name,
            "vnf_list": self.build_chain(player=player, counter=counter),
            "bandwidth": bandwidth,
            "src_node": src_node,
            "dst_node": dst_node,
            "closer_router": closer_router,
            "latency": self.latency_request,
            "duration": duration,
            "service_type": self.name,
        }


class ServiceCatalog:
    """Registro central de tipos de serviço disponíveis na simulação."""

    def __init__(self):
        self._services: dict[str, ServiceTemplate] = {}

    def register(self, template: ServiceTemplate) -> None:
        self._services[template.name] = template

    def get(self, name: str) -> ServiceTemplate:
        return self._services[name]

    def list(self) -> list[str]:
        return list(self._services.keys())

    def weighted_choice(self, rng=random, weights: dict[str, float] | None = None):
        """Sorteia um template respeitando os pesos (default: uniforme)."""
        if weights is None:
            names = self.list()
            return self._services[rng.choice(names)]
        # FIX: considera apenas os serviços presentes nos pesos (o mix configurado),
        # evitando KeyError quando o mix é um subconjunto do catálogo (ex.: só "muar").
        names = [n for n in weights if n in self._services]
        if not names:
            raise ValueError(
                f"Nenhum serviço de {list(weights)} está registrado no catálogo. "
                f"Disponíveis: {self.list()}"
            )
        probs = [weights[n] for n in names]
        return self._services[rng.choices(names, weights=probs, k=1)[0]]


def build_default_catalog() -> ServiceCatalog:
    """Popula o catálogo com os serviços de exemplo (inclui o MUAR)."""
    catalog = ServiceCatalog()

    # --- MUAR (cenário original; mantém compatibilidade) ---
    muar_chain = [
        {"type": VNFType.TYPE1, "name": "IA_DET_FT_{counter}", "CPU": 22.0, "cache": 0,
         "in_bw": 150.0, "out_bw": 151.0, "latency": 0.3},
        {"type": VNFType.TYPE1, "name": "MA_region_{closer}", "CPU": 10.0, "cache": 80.0,
         "in_bw": 151.0, "out_bw": 80.0, "latency": 0.4},
        {"type": VNFType.TYPE1, "name": "RE_region_{closer}", "CPU": 8.0, "cache": 0,
         "in_bw": 80.0, "out_bw": 64.0, "latency": 0.5},
        {"type": VNFType.TYPE1, "name": "EC_TC_{counter}", "CPU": 7.0, "cache": 0,
         "in_bw": 64.0, "out_bw": 60.0, "latency": 0.4},
    ]
    catalog.register(ServiceTemplate(
        name="muar",
        latency_request=1000.0,
        vnf_chain=muar_chain,
        shareable=True,
        shareable_prefixes=SHAREABLE_PREFIXES,
    ))

    # --- Streaming de vídeo (CPU/banda altos, VNFs que transformam tráfego) ---
    streaming_chain = [
        {"type": VNFType.TYPE1, "name": "packetizer_{player}_{counter}", "CPU": 15.0, "cache": 0,
         "in_bw": 40.0, "out_bw": 42.0, "latency": 0.2},
        {"type": VNFType.TYPE2, "name": "encoder_{player}_{counter}", "CPU": 45.0, "cache": 0,
         "in_bw": 42.0, "out_bw": 42.0, "latency": 1.2, "params": {"factor": 1.4}},
        {"type": VNFType.TYPE3, "name": "transcoder_{player}_{counter}", "CPU": 35.0, "cache": 0,
         "in_bw": 42.0, "out_bw": 42.0, "latency": 1.0, "params": {"factor": 0.7}},
    ]
    catalog.register(ServiceTemplate(
        name="streaming",
        latency_request=150.0,
        vnf_chain=streaming_chain,
        shareable=False,
    ))

    # --- VoIP (latência estrita, baixa CPU/banda) ---
    voip_chain = [
        {"type": VNFType.TYPE1, "name": "jitter_buf_{player}_{counter}", "CPU": 5.0, "cache": 0,
         "in_bw": 0.1, "out_bw": 0.1, "latency": 0.1},
        {"type": VNFType.TYPE1, "name": "codec_{player}_{counter}", "CPU": 8.0, "cache": 0,
         "in_bw": 0.1, "out_bw": 0.1, "latency": 0.3},
    ]
    catalog.register(ServiceTemplate(
        name="voip",
        latency_request=60.0,
        vnf_chain=voip_chain,
        shareable=False,
    ))

    # --- IoT / telemetria (agregação) ---
    iot_chain = [
        {"type": VNFType.TYPE1, "name": "ingest_{player}_{counter}", "CPU": 2.0, "cache": 0,
         "in_bw": 1.0, "out_bw": 1.0, "latency": 0.2},
        {"type": VNFType.TYPE4, "name": "aggregator_{player}_{counter}", "CPU": 6.0, "cache": 0,
         "in_bw": 1.0, "out_bw": 1.0, "latency": 0.4},
    ]
    catalog.register(ServiceTemplate(
        name="iot",
        latency_request=200.0,
        vnf_chain=iot_chain,
        shareable=False,
    ))

    return catalog

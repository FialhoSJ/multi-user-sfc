from dataclasses import dataclass, field


@dataclass
class Parameter:
    """Defines the simulation's constant declarations."""

    ma_bw: float
    ma: float
    uni_bw: float
    uni: float
    re_bw: float
    re: float
    ec_tc_bw: float
    ec_tc: float
    det: float
    ia_det_ft: float
    ia_det_ft_bw: float
    ma: float
    uni: float
    re: float
    ec_tc: float
    mono: float
    total: float
    n_sessions: int
    ft: float

    ia: float = field(default=0)
    number_of_nodes: int = field(default=36)
    src_node: int = field(default=0)
    max_duration: int = field(default=120)
    latency: float = field(default=6)
    fator: float = field(default=0.25)
    average_time_session_arrival: float = field(default=10)
    ia_bw: float = field(default=150)
    cpb: float = field(default=10e6)
    ca_size: float = field(default=240)
    chr: float = field(default=1 / 3)
    ft_bw: float = field(default=0.144)
    det_bw: float = field(default=0.230)

    def __post__init__(self):
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


@dataclass
class MUVRParam:
    pass


@dataclass
class MUARParam:
    pass

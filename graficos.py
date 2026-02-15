# %%
# %% [Célula 1] Imports e Configurações
import os
import glob
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from typing import List, Dict, Optional, Tuple, Sequence

# Configurações de avisos e estilo
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

# Configuração global do Matplotlib (mantendo o estilo original)
plt.rcParams.update({
    'font.size': 25,
    'axes.labelsize': 27,
    'legend.fontsize': 15,
    'xtick.labelsize': 22,
    'ytick.labelsize': 22,
    'axes.titlesize': 27,
    'hatch.linewidth': 0.5
})

# %%
# %% [Célula 2] Configurações e Constantes

# Diretório base dos resultados
RESULTS_DIR = '../../results/results_flows*/*'

# Métricas que serão lidas ou calculadas
METRICAS_COLETA = [
    'tempo', 'cpu_utilization', 'gpu_utilization', 'cache_utilization',
    'bandwidth_utilization', 'bandwidth_utilization_per_flow',
    'success', 'latency', 'decision_time_ms', 'running_sfcs',
    'cpu_saved', 'shared_vnfs', 'shared_vnfs_per_flow',
    'cpu_per_flow', 'gpu_per_flow', 'cache_per_flow',
    "total_energy_consumption", "energy_consumption_per_flow",
    "jain_fairness_latencia_por_sessao", "acceptance_rate",
    "queue_time", "jain_cpu", "jain_gpu", "jain_bw", "avg_sfc_reliability"
]

# Mapeamento: Nome na Pasta -> Nome no Gráfico
NOME_ALGORITMOS_MAP = {
    'g': 'Greedy',
    'ga': 'OSCIM',
    'msf': 'MSF',
    'goku': 'OSCIM',
    'vegeta': 'Resilient-OSCIM',
    'musfico': 'MuSFiCO',
    'greedyb': 'GreedyB',
    'kuririnPPO': 'INOMMUS',
    'kuririnMaskablePPO': 'INOMMUS',
    'darsppo': 'DARSPPO',
    'darsppoMaskablePPO': 'DA-RSPPO',
    'hephaestus': 'hephaestus',
    'hephaestusMaskablePPO': 'Hephaestus',
    'REPLICMASKABLEPPO': 'RELIC',
    'DRLMASKABLEPPO': 'REINFORCE'
}

# Cores específicas para cada algoritmo (Consistência visual)
CORES_ALGORITMOS = {
    "INOMMUS": "#1f77b4",     # Azul
    "DA-RSPPO": "#ff7f0e",# Laranja
    "RELIC": "#ddff00",        
    "Hephaestus": "#2ca02c",  # Verde
    "OSCIM": "#d62728",       # Vermelho
    "GreedyB": "#9467bd",     # Roxo
    "Greedy": "#8c564b",       # Marrom
}

# Padrões de hachura (textura das barras)
HATCH_PATTERNS = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

# %%
# %% [Célula 3] Funções de Processamento de Dados

def jain_fairness_index(x):
    """Calcula o índice de justiça de Jain para uma série."""
    x = x.dropna()
    n = len(x)
    if n == 0: return np.nan
    sum_sq = (x * x).sum()
    if sum_sq == 0: return 1.0
    return (x.sum() ** 2) / (n * sum_sq)

def enriquecer_dataframe(df):
    """Adiciona colunas calculadas ao DataFrame."""
    df = df.copy()

    # 1. Jain Fairness por Sessão
    if 'sfc_id' in df.columns and 'latency' in df.columns:
        df['session_id'] = df['sfc_id'].str.split('_').str[-1]
        df['jain_fairness_latencia_por_sessao'] = df.groupby('session_id')['latency'].transform(jain_fairness_index)
        df['jain_fairness_latencia_por_sessao'] = df['jain_fairness_latencia_por_sessao'].fillna(0)
        df.drop(columns=['session_id'], inplace=True)

    # 2. Métricas "Por Fluxo" (Divisão segura por running_sfcs)
    for metric, new_col in [
        ('total_energy_consumption', 'energy_consumption_per_flow'),
        ('bandwidth_utilization', 'bandwidth_utilization_per_flow'),
        ('shared_vnfs', 'shared_vnfs_per_flow')
    ]:
        if metric in df.columns:
            df[new_col] = np.where(df['running_sfcs'] == 0, 0.0, df[metric] / df['running_sfcs'])

    # 3. QoS Score
    if 'latency' in df.columns and 'success' in df.columns:
        df['pontuacao_latencia'] = df['latency'].apply(lambda x: 0 if pd.isna(x) else (-1 if x > 6 else 2))
        df['pontuacao_success'] = df['success'] * 10
        df['QoS'] = df['pontuacao_latencia'] + df['pontuacao_success']
    
    return df

# %%
# %% [Célula 4] Coleta de Dados

def coletar_dados(diretorio_pattern):
    data_frames = []
    diretorios = glob.glob(diretorio_pattern)
    
    print(f"Encontrados {len(diretorios)} diretórios de simulação.")

    for alg_dir in diretorios:
        # Extrair nome do algoritmo do caminho
        dir_name = os.path.basename(os.path.normpath(alg_dir))
        raw_name = dir_name.split('_')[0]
        # Tratamento para caminhos Windows/Linux misturados
        if '\\' in raw_name: raw_name = raw_name.split('\\')[-1]
        
        alg_name = NOME_ALGORITMOS_MAP.get(raw_name, raw_name)
        
        files = glob.glob(os.path.join(alg_dir, "*.csv"))
        # print(f"Lendo {alg_name}: {len(files)} arquivos.")

        for file in files:
            try:
                df = pd.read_csv(file)
                
                # Ajustes iniciais e Filtros
                df = enriquecer_dataframe(df)
                
                # Calcular tempo relativo (segundos desde o início)
                start_time = df['timestamp'].values[0]
                df['tempo'] = (df['timestamp'] - start_time).astype(int)
                
                # Verificar se a simulação foi válida (> 100s)
                if df['tempo'].max() < 100:
                    continue

                # Sucesso acumulativo (média expandida)
                df['success'] = df['success'].expanding().mean()

                # Agrupamento temporal (segundo a segundo) e preenchimento (ffill)
                df_grouped = df.groupby('tempo', as_index=False).mean(numeric_only=True)
                max_time = df_grouped['tempo'].max()
                
                # Reindexar para garantir continuidade temporal
                idx_completo = pd.RangeIndex(start=0, stop=max_time + 1, name='tempo')
                df_final = df_grouped.set_index('tempo').reindex(idx_completo).ffill().reset_index()
                
                # Recorte de 1000 segundos (padrão do paper)
                df_final = df_final.iloc[0:1000]
                df_final['algorithm'] = alg_name
                
                data_frames.append(df_final)

            except Exception as e:
                print(f"Erro ao ler {file}: {e}")
                continue

    if data_frames:
        big_df = pd.concat(data_frames, ignore_index=True)
        # Normalizações globais finais (/100)
        cols_to_normalize = ['acceptance_rate', 'cpu_per_flow', 'gpu_per_flow', 'cache_per_flow']
        for col in cols_to_normalize:
            if col in big_df.columns:
                big_df[col] = big_df[col] / 100
        return big_df
    else:
        return pd.DataFrame()

# %%
# %% [Célula 5] Executar Carregamento

print("Iniciando coleta...")
big_data = coletar_dados(RESULTS_DIR)

# Filtro final: Remover período de aquecimento (< 200s)
if not big_data.empty:
    print(f"Total de linhas carregadas: {len(big_data)}")
    big_data = big_data[big_data['tempo'] >= 0].copy()
    print(f"Linhas após filtro de aquecimento (>= 200s): {len(big_data)}")
else:
    print("Nenhum dado encontrado! Verifique o caminho RESULTS_DIR.")

# %%
# %% [Célula 6] Funções de Plotagem

def create_boxplot(df, y_col, title, filename=None, show_outliers=False, steps=200):
    """
    Cria boxplot temporal agrupado por janelas de tempo.
    """
    # Preparar dados
    df_plot = df.copy()
    df_plot['window'] = (df_plot['tempo'] // steps + 1) * steps
    
    # Ordenação e Labels
    windows = sorted(df_plot['window'].unique())
    algs = sorted(df_plot['algorithm'].unique(), key=lambda x: list(CORES_ALGORITMOS.keys()).index(x) if x in CORES_ALGORITMOS else 999)
    
    # Configurar Figura
    fig, ax = plt.subplots(figsize=(9.8, 6.2)) # Aproximadamente 950x600px
    
    group_width = 0.8
    box_width = group_width / len(algs)
    positions_base = np.arange(len(windows))
    
    legend_handles = []

    for i, alg in enumerate(algs):
        color = CORES_ALGORITMOS.get(alg, plt.cm.tab10(i))
        hatch = HATCH_PATTERNS[i % len(HATCH_PATTERNS)]
        
        # Filtrar dados do algoritmo
        alg_data = df_plot[df_plot['algorithm'] == alg]
        
        # Dados por janela
        data_per_window = []
        for w in windows:
            vals = alg_data[alg_data['window'] == w][y_col].dropna().values
            data_per_window.append(vals)
            
        # Posição deslocada para agrupar
        offset = (i - len(algs)/2) * box_width + box_width/2
        positions = positions_base + offset
        
        # Estilo dos Outliers
        flier_props = dict(marker='o', markersize=3, markerfacecolor=color, markeredgecolor=color, alpha=0.7)
        
        # Desenhar Boxplot
        bp = ax.boxplot(data_per_window, positions=positions, widths=box_width*0.9,
                        patch_artist=True, showfliers=show_outliers,
                        boxprops=dict(facecolor=color, hatch=hatch, alpha=0.5, linewidth=1.2),
                        medianprops=dict(color='black', linewidth=1.8),
                        flierprops=flier_props if show_outliers else dict(marker=''))

        # Reforçar cores e hatch
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_hatch(hatch)
            patch.set_edgecolor('black')

        legend_handles.append(Patch(facecolor=color, edgecolor='black', hatch=hatch, alpha=0.5, label=alg))

    # Formatação do Eixo
    ax.set_ylabel(title, labelpad=10)
    ax.set_xticks(positions_base)
    ax.set_xticklabels([]) # Remove labels do eixo X conforme solicitado
    ax.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
    ax.grid(axis='y', linestyle='--', alpha=0.5, linewidth=0.8)

    # Legenda
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=min(4, len(algs)), frameon=False, handlelength=2.2)

    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300)
        print(f"Salvo: {filename}")
    plt.show()

def create_binned_boxplot(df, x_col, y_col, x_label, y_label, bin_size=10, filename=None):
    """
    Cria boxplot agrupado por bins de uma métrica X.
    """
    df_plot = df.copy()
    
    # Criar Bins
    # Multiplica por 100 assumindo que x_col está normalizado (0-1), ajuste se necessário
    df_plot['x_scaled'] = df_plot[x_col] * 100
    df_plot['bin'] = (np.ceil(df_plot['x_scaled'] / bin_size) * bin_size).astype(int)
    
    bins = sorted(df_plot['bin'].unique())
    algs = sorted(df_plot['algorithm'].unique()) # Ou ordem customizada
    
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    
    box_width = 0.8 / len(algs)
    positions_base = np.arange(len(bins))
    legend_handles = []

    for i, alg in enumerate(algs):
        color = CORES_ALGORITMOS.get(alg, plt.cm.tab10(i))
        hatch = HATCH_PATTERNS[i % len(HATCH_PATTERNS)]
        
        alg_data = df_plot[df_plot['algorithm'] == alg]
        data_per_bin = [alg_data[alg_data['bin'] == b][y_col].dropna().values for b in bins]
        
        offset = (i - len(algs)/2) * box_width + box_width/2
        bp = ax.boxplot(data_per_bin, positions=positions_base + offset, widths=box_width*0.9,
                        patch_artist=True, showfliers=True,
                        boxprops=dict(facecolor=color, hatch=hatch, alpha=0.5),
                        medianprops=dict(color='black'),
                        flierprops=dict(marker='o', markersize=3, markerfacecolor=color, alpha=0.7))
        
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_hatch(hatch)
            
        legend_handles.append(Patch(facecolor=color, hatch=hatch, alpha=0.5, label=alg))

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xticks(positions_base)
    ax.set_xticklabels([str(b) for b in bins])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=min(4, len(algs)), frameon=True)
    
    plt.tight_layout()
    if filename: plt.savefig(filename, dpi=300)
    plt.show()

# %%
# %% [Célula 7] Execução - Gráficos Temporais

# Seleção de Algoritmos para Plotar
algoritmos_foco = ['RELIC', 'REINFORCE']
df_foco = big_data[big_data['algorithm'].isin(algoritmos_foco)].copy()

# Mapeamento Métrica -> Título do Eixo Y
metricas_plot = {
    "Latency (ms)": "latency",
    "Acceptance Ratio (%)": "acceptance_rate",
    "Decision Time (ms)": "decision_time_ms",
    "CPU Util. per Flow (%)": "cpu_per_flow",
    "GPU Util. per Flow (%)": "gpu_per_flow",
    "Band. Util. per Flow (%)": "bandwidth_utilization_per_flow",
    "Power Co. (W) per Flow": "energy_consumption_per_flow",
    "Service Lat. Equality": "jain_fairness_latencia_por_sessao",
    "avg_sfc_reliability": "avg_sfc_reliability",
}

for titulo, col in metricas_plot.items():
    if col not in df_foco.columns:
        print(f"Aviso: Coluna {col} não encontrada.")
        continue
        
    # Multiplicador para porcentagem (se necessário)
    # No ETL já dividimos por 100, então se o título tem %, multiplicamos de volta por 100 para o plot
    fator = 100.0 if "%" in titulo else 1.0
    
    # Criar coluna temporária para plotagem
    col_plot = f"{col}_plot"
    df_foco[col_plot] = df_foco[col] * fator
    
    # Nome do arquivo seguro
    safe_name = re.sub(r"[^\w\-]+", "_", titulo)
    
    # Gerar duas versões: com e sem outliers
    create_boxplot(df_foco, col_plot, titulo, filename=f"boxplot_{safe_name}_clean.pdf", show_outliers=False)
    create_boxplot(df_foco, col_plot, titulo, filename=f"boxplot_{safe_name}_outliers.pdf", show_outliers=True)

# %%
# %% [Célula 8] Execução - Gráfico Binado

create_binned_boxplot(
    df=df_foco,
    x_col='bandwidth_utilization',
    y_col='acceptance_rate', # Já está em decimal no DF, mas a função multiplica por 100 para bin
    x_label='Bandwidth Utilization (%)',
    y_label='Acceptance Ratio (%)',
    bin_size=10,
    filename="binned_acceptance_vs_bw.pdf"
)

# %%
# %% [Célula Atualizada] Dashboard de Risco com Suavização (Rolling Mean)

# ==========================================
# CONFIGURAÇÃO DE SUAVIZAÇÃO
# ==========================================
# Define quantos segundos de dados são usados para calcular a média.
# Valor maior = linha mais lisa (mas pode esconder picos rápidos).
# Valor menor = linha mais detalhada (mas mais tremida).
JANELA_SUAVIZACAO = 75
# ==========================================

def plot_risk_dashboard_suavizado(df, janela=20):
    # Configuração do painel 2x2
    fig, axs = plt.subplots(2, 2, figsize=(18, 12), sharex=True)
    
    plots_config = [
        {
            "ax": axs[0, 0],
            "col": "running_sfcs",
            "title": "Total Running SFCs (All Levels)",
            "ylabel": "Number of SFCs"
        },
        {
            "ax": axs[0, 1],
            "col": "high_risk_sfcs",
            "title": "High Risk SFCs",
            "ylabel": "Count"
        },
        {
            "ax": axs[1, 0],
            "col": "medium_risk_sfcs",
            "title": "Medium Risk SFCs",
            "ylabel": "Count"
        },
        {
            "ax": axs[1, 1],
            "col": "low_risk_sfcs",
            "title": "Low Risk SFCs",
            "ylabel": "Count"
        }
    ]

    # Ordenar algoritmos para consistência na legenda
    algoritmos = sorted(df['algorithm'].unique(), 
                        key=lambda x: list(CORES_ALGORITMOS.keys()).index(x) if x in CORES_ALGORITMOS else 999)

    alg_avaliados = ['RELIC', 'REINFORCE']  # Foco nos algoritmos de interesse
    for config in plots_config:
        ax = config["ax"]
        col = config["col"]
        
        for alg in algoritmos:
            if alg not in alg_avaliados:
                continue
            df_alg = df[df['algorithm'] == alg]
            if df_alg.empty: continue
            
            # 1. Agrupar dados brutos por tempo
            series_raw = df_alg.groupby('tempo')[col].mean()
            
            # 2. APLICAR SUAVIZAÇÃO (ROLLING MEAN)
            # min_periods=1 garante que não fiquem buracos no começo do gráfico
            if janela > 1:
                series_plot = series_raw.rolling(window=janela, min_periods=1).mean()
            else:
                series_plot = series_raw
            
            color = CORES_ALGORITMOS.get(alg, 'gray')
            
            # Plotar linha suavizada
            # Aumentei o linewidth para 2.5 para ficar mais bonito
            ax.plot(series_plot.index, series_plot.values, label=alg, color=color, linewidth=2.5, alpha=0.9)

        # Estilização
        ax.set_xlim(series_plot.index.min(), series_plot.index.max()*0.9)
        ax.set_title(config["title"], pad=10, fontweight='bold', fontsize=18)
        ax.set_ylabel(config["ylabel"], fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        if config["ax"] in axs[1, :]:
            ax.set_xlabel("Time (s)", fontsize=16)

    # Legenda global
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), 
               ncol=len(algoritmos), frameon=False, fontsize=16)

    plt.tight_layout()
    plt.savefig(f"risk_dashboard_smooth_{janela}.pdf", dpi=300, bbox_inches='tight')
    plt.show()

# Executar
if 'df_risco' in locals() and not df_risco.empty:
    plot_risk_dashboard_suavizado(df_risco, janela=JANELA_SUAVIZACAO)
else:
    # Caso você tenha reiniciado o kernel, recarrega o df_risco
    # Certifique-se que big_data está carregado
    df_risco = big_data[big_data['tempo'] >= 0].copy()
    plot_risk_dashboard_suavizado(df_risco, janela=JANELA_SUAVIZACAO)

# %%
# %% [Célula de Ajuste Fino] Gráfico de Confiabilidade (Título Limpo)

def plot_reliability_final(df, janela=20):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Ordenação dos algoritmos
    algoritmos = sorted(
        df['algorithm'].unique(), 
        key=lambda x: list(CORES_ALGORITMOS.keys()).index(x) if x in CORES_ALGORITMOS else 999
    )

    min_val_global = 1.0
    max_val_global = 0.0

    for alg in algoritmos:
        df_alg = df[df['algorithm'] == alg]
        if df_alg.empty:
            continue
        
        series_raw = df_alg.groupby('tempo')['avg_sfc_reliability'].mean()
        
        if janela > 1:
            series_plot = series_raw.rolling(window=janela, min_periods=1).mean()
        else:
            series_plot = series_raw
            
        min_val_global = min(min_val_global, series_plot.min())
        max_val_global = max(max_val_global, series_plot.max())
        
        color = CORES_ALGORITMOS.get(alg, 'gray')
        ax.plot(
            series_plot.index,
            series_plot.values,
            label=alg,
            color=color,
            linewidth=2.5,
            alpha=0.9
        )

    # --- TÍTULO E LEGENDAS (CORRIGIDO) ---
    ax.set_xlim(series_plot.index.min(), series_plot.index.max())
    ax.set_title("Average SFC Reliability", pad=18, fontweight='bold', fontsize=20)
    ax.set_ylabel("Avg. Reliability", fontsize=18)
    ax.set_xlabel("Time (s)", fontsize=18)
    
    # Legenda ajustada para não colidir com o título
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(algoritmos),
        frameon=False,
        fontsize=16
    )

    margin = (max_val_global - min_val_global) * 0.1
    y_lower = max(0, min_val_global - margin)
    y_upper = min(1.0005, max_val_global + margin)
    ax.set_ylim(y_lower, y_upper)

    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.tick_params(axis='both', labelsize=15)
    
    plt.tight_layout()
    plt.savefig("avg_reliability_final.pdf", dpi=300, bbox_inches='tight')
    plt.show()

plot_reliability_final(df_foco, janela=200)

# %%
# %% [Célula Nova] Gráfico de Falhas por Nível de Risco (Resiliência)

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ==========================================
# 1. CONFIGURAÇÃO
# ==========================================
# Caminho para a pasta results_resilient.
# Ajuste o caminho relativo conforme onde seu notebook está salvo.
# Se estiver na raiz: 'results/results_resilient/*'
# Se estiver em subpasta: '../../results/results_resilient/*'
DIR_RESILIENT = '../../results/results_resilient/*' 

# Mapeamento de nomes (igual ao anterior para consistência)
MAPA_NOMES = {
    'ga': 'OSCIM',
    'greedyb': 'GreedyB',
    'kuririn': 'INOMMUS',
    'darsppo': 'DA-RSPPO',
    'hephaestus': 'Hephaestus',
    'REPLICMASKABLEPPO': 'RELIC',
    'DRLMASKABLEPPO': 'REINFORCE',
}

# Cores e Hachuras (Estilo consistente)
CORES = {
    "INOMMUS": "#1f77b4",     # Azul
    "DA-RSPPO": "#ff7f0e",    # Laranja
    "Hephaestus": "#2ca02c",  # Verde
    'REPIC': "#ddff00",        # Amarelo
    'REINFORCE': "#ff00ff",      # Magenta
    "OSCIM": "#d62728",       # Vermelho
    "GreedyB": "#9467bd",     # Roxo
}
HACHURAS = ['//', '\\\\', '||', '--', '++', 'xx', 'oo', 'OO', '..', '**']

# ==========================================
# 2. PROCESSAMENTO DOS DADOS
# ==========================================
def carregar_dados_resiliencia(diretorio_pattern):
    # {Algoritmo: {'Low': [], 'Medium': [], 'High': []}}
    dados_agrupados = {}
    
    pastas = glob.glob(diretorio_pattern)
    print(f"Pastas encontradas em {diretorio_pattern}: {len(pastas)}")
    
    for pasta in pastas:
        nome_pasta = os.path.basename(os.path.normpath(pasta))
        prefixo = nome_pasta.split('_')[0]
        nome_alg = next((v for k, v in MAPA_NOMES.items() if k in prefixo), prefixo)
        
        if nome_alg not in dados_agrupados:
            dados_agrupados[nome_alg] = {'Low': [], 'Medium': [], 'High': []}
            
        csvs = glob.glob(os.path.join(pasta, "*.csv"))
        for arquivo in csvs:
            try:
                df = pd.read_csv(arquivo)
                if 'affected_by_low_risk_node' in df.columns:
                    dados_agrupados[nome_alg]['Low'].append(df['affected_by_low_risk_node'].sum())
                    dados_agrupados[nome_alg]['Medium'].append(df['affected_by_med_risk_node'].sum())
                    dados_agrupados[nome_alg]['High'].append(df['affected_by_high_risk_node'].sum())
            except Exception as e:
                print(f"Erro ao ler {arquivo}: {e}")

    # Calcula a média final
    dados_medios = {}
    for alg, riscos in dados_agrupados.items():
        dados_medios[alg] = {
            risco: (np.mean(valores) if valores else 0)
            for risco, valores in riscos.items()
        }

    return dados_medios


# ==========================================
# 3. PLOTAGEM
# ==========================================
def plot_failures_by_risk(dados):
    if not dados:
        print("Nenhum dado encontrado para plotar.")
        return

    # Categorias e Algoritmos
    categorias = ['Low Risk', 'Medium Risk', 'High Risk']
    mapa_chaves_csv = {'Low Risk': 'Low', 'Medium Risk': 'Medium', 'High Risk': 'High'}
    
    # Ordenar algoritmos (INOMMUS primeiro se existir, ou ordem alfabética)
    algoritmos = sorted(dados.keys(), key=lambda x: list(CORES.keys()).index(x) if x in CORES else 999)
    
    # Configuração do Gráfico
    x = np.arange(len(categorias))
    width = 0.8 / len(algoritmos)  # Largura da barra
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Loop para criar as barras de cada algoritmo
    for i, alg in enumerate(algoritmos):
        # Extrai os valores na ordem: Low, Medium, High
        valores = [dados[alg][mapa_chaves_csv[cat]] for cat in categorias]
        
        posicao = x + (i - len(algoritmos)/2) * width + width/2
        cor = CORES.get(alg, 'gray')
        hachura = HACHURAS[i % len(HACHURAS)]
        
        barras = ax.bar(posicao, valores, width, label=alg, 
                        color=cor, hatch=hachura, edgecolor='black', alpha=0.7)
        
        # Opcional: Adicionar o valor em cima da barra
        # ax.bar_label(barras, padding=3, fmt='%d', fontsize=10)

    # Estilização
    ax.set_ylabel('Total Failed SFCs', fontsize=16)
    ax.set_title('Total Failures by Risk Level', fontsize=18, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias, fontsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legenda
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), 
              ncol=len(algoritmos), frameon=False, fontsize=14)
    
    plt.tight_layout()
    plt.savefig('failures_by_risk_grouped.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# ==========================================
# 4. EXECUÇÃO
# ==========================================
print("Carregando dados de resiliência...")
dados_resiliencia = carregar_dados_resiliencia(DIR_RESILIENT)
plot_failures_by_risk(dados_resiliencia)

# %%
def carregar_dados_resiliencia_completo(diretorio_pattern):
    """
    Lê duas fontes de dados:
    1. df_impact (Macro): Totais de falhas, riscos e médias globais.
    2. df_resil (Micro): Detalhes de cada SFC (latência, tipo de recuperação).
    """
    list_impact = []
    list_resil = []
    
    diretorios = glob.glob(diretorio_pattern)
    print(f"Lendo resiliência de {len(diretorios)} diretórios...")

    for alg_dir in diretorios:
        # Extração do nome do algoritmo
        dir_name = os.path.basename(os.path.normpath(alg_dir))
        raw_name = dir_name.split('_')[0]
        alg_name = NOME_ALGORITMOS_MAP.get(raw_name, raw_name)
        
        # 1. Ler Crash Impact (Macro) - JÁ ESTAVA CERTO
        files_impact = glob.glob(os.path.join(alg_dir, "crash_impact_*.csv"))
        for f in files_impact:
            try:
                df = pd.read_csv(f)
                df['algorithm'] = alg_name
                list_impact.append(df)
            except Exception as e:
                print(f"Erro no impact {alg_name}: {e}")

        # 2. Ler Resilient Results (Micro) - CORRIGIDO AQUI
        # Mudamos de um arquivo fixo para um padrão com glob (*.csv)
        files_resil = glob.glob(os.path.join(alg_dir, "resilient_results_*.csv"))
        
        for f in files_resil:
            try:
                df_r = pd.read_csv(f)
                df_r['algorithm'] = alg_name
                list_resil.append(df_r)
            except Exception as e:
                print(f"Erro no resil {alg_name}: {e}")

    df_impact_final = pd.concat(list_impact, ignore_index=True) if list_impact else pd.DataFrame()
    df_resil_final = pd.concat(list_resil, ignore_index=True) if list_resil else pd.DataFrame()
    
    return df_impact_final, df_resil_final




PATH_RESILIENT = '../../results/results_resilient/*'

print("Carregando dados de resiliência...")
df_impacto, df_detalhado = carregar_dados_resiliencia_completo(PATH_RESILIENT)

print(f"Dados Macro carregados: {len(df_impacto)} linhas")
print(f"Dados Micro carregados: {len(df_detalhado)} linhas")

# %%
def plot_failures_risk_stacked(df_impact):
    if df_impact.empty: return

    # --- CORREÇÃO 1: Garantir que as colunas sejam numéricas ---
    cols_to_fix = ['affected_by_low_risk_node', 'affected_by_med_risk_node', 'affected_by_high_risk_node']
    for col in cols_to_fix:
        # Tenta converter para número, se der erro (ex: texto), transforma em NaN e depois em 0
        if col in df_impact.columns:
            df_impact[col] = pd.to_numeric(df_impact[col], errors='coerce').fillna(0)

    # Agrupar médias por algoritmo
    df_agg = df_impact.groupby('algorithm')[cols_to_fix].mean()
    
    # Ordenação baseada nas cores definidas
    algs_sorted = sorted(df_agg.index, key=lambda x: list(CORES_ALGORITMOS.keys()).index(x) if x in CORES_ALGORITMOS else 999)
    df_agg = df_agg.loc[algs_sorted]

    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Configuração das Barras Empilhadas
    bar_width = 0.6
    indices = np.arange(len(df_agg))
    
    colors_risk = ['#2ca02c', '#ff7f0e', '#d62728'] # Verde, Laranja, Vermelho
    labels_risk = ['Low Risk', 'Medium Risk', 'High Risk']
    
    # Inicializa o fundo (bottom) como float
    bottoms = np.zeros(len(df_agg), dtype=float)
    
    for i, col in enumerate(cols_to_fix):
        # --- CORREÇÃO 2: Garantir que values seja array de floats ---
        values = df_agg[col].values.astype(float)
        
        ax.bar(indices, values, bar_width, bottom=bottoms, 
               label=labels_risk[i], color=colors_risk[i], 
               edgecolor='black', alpha=0.85)
        
        # Agora a soma vai funcionar porque ambos são floats
        bottoms += values

    # Estilização
    ax.set_ylabel('Avg. Affected SFCs per Crash')
    ax.set_title('Impact Profile: Affected SFCs by Risk Level', pad=20, fontsize=20)
    ax.set_xticks(indices)
    ax.set_xticklabels(df_agg.index, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    
    # Legenda
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
              ncol=3, frameon=False, fontsize=16)

    plt.tight_layout()
    plt.savefig('resilience_risk_profile.pdf', dpi=300)
    plt.show()

plot_failures_risk_stacked(df_impacto)

# %%
def plot_latency_degradation_box(df_resil):
    if df_resil.empty: return
    
    # Filtra apenas quem recuperou (senão não tem latência final)
    df_plot = df_resil[df_resil['recover_success'] == True].copy()
    
    # Ordenação
    algs = sorted(df_plot['algorithm'].unique(), 
                  key=lambda x: list(CORES_ALGORITMOS.keys()).index(x) if x in CORES_ALGORITMOS else 999)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_to_plot = []
    for alg in algs:
        # latency_diff = Nova - Antiga. 
        # Valores altos significam que piorou muito.
        vals = df_plot[df_plot['algorithm'] == alg]['latency_diff'].dropna().values
        data_to_plot.append(vals)
        
    # Plotagem
    positions = np.arange(len(algs))
    
    for i, (alg, data) in enumerate(zip(algs, data_to_plot)):
        color = CORES_ALGORITMOS.get(alg, 'gray')
        hatch = HATCH_PATTERNS[i % len(HATCH_PATTERNS)]
        
        bp = ax.boxplot(data, positions=[i], widths=0.6, patch_artist=True,
                        boxprops=dict(facecolor=color, hatch=hatch, alpha=0.6),
                        medianprops=dict(color='black', linewidth=2),
                        flierprops=dict(marker='o', markerfacecolor=color, alpha=0.5))
        
        # Correção visual do hatch
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_hatch(hatch)

    ax.set_ylabel('Latency Increase (ms)')
    ax.set_title('Performance Penalty After Recovery', pad=20, fontsize=20)
    ax.set_xticks(positions)
    ax.set_xticklabels(algs)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Linha de referência no zero (sem impacto)
    ax.axhline(0, color='black', linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig('resilience_latency_penalty.pdf', dpi=300)
    plt.show()

plot_latency_degradation_box(df_detalhado)

# %%
def plot_recovery_mechanism(df_resil):
    if df_resil.empty: return

    # Agrupa por algoritmo e conta tipos de sucesso
    # backup_success = True -> Usou réplica (Rápido)
    # recover_success = True AND backup_success = False -> Reinstanciou (Lento)
    # recover_success = False -> Falhou Total
    
    resumo = []
    algs = df_resil['algorithm'].unique()
    
    for alg in algs:
        d = df_resil[df_resil['algorithm'] == alg]
        total = len(d)
        if total == 0: continue
        
        backup = d['backup_success'].sum()
        # Redeploy é quem recuperou MAS não foi backup
        redeploy = len(d[(d['recover_success'] == True) & (d['backup_success'] == False)])
        failed = len(d[d['recover_success'] == False])
        
        resumo.append({
            'Algorithm': alg,
            'Backup (Fast)': (backup / total) * 100,
            'Redeploy (Slow)': (redeploy / total) * 100,
            'Failed': (failed / total) * 100
        })
        
    df_plot = pd.DataFrame(resumo).set_index('Algorithm')
    # Ordenar
    algs_sorted = sorted(df_plot.index, key=lambda x: list(CORES_ALGORITMOS.keys()).index(x) if x in CORES_ALGORITMOS else 999)
    df_plot = df_plot.loc[algs_sorted]

    # Plot
    fig, ax = plt.subplots(figsize=(14, 14))
    
    # Cores customizadas para o mecanismo
    colors_mech = ['#1f77b4', '#ff7f0e', '#7f7f7f'] # Azul, Laranja, Cinza
    
    df_plot.plot(kind='bar', stacked=True, color=colors_mech, ax=ax, edgecolor='black', width=0.7)
    
    ax.set_ylabel('Percentage of Affected SFCs (%)')
    ax.set_title('Resilience Mechanism Efficiency', pad=20, fontsize=20)
    ax.set_xlabel('')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=14)
    
    # Rótulos nas barras
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('resilience_mechanism_breakdown.pdf', dpi=300)
    plt.show()

plot_recovery_mechanism(df_detalhado)

# %%
# %%
# %% [Célula Nova] Gráfico de Resiliência Detalhado (Baseado nas Novas Colunas)

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from matplotlib.patches import Patch

# ==========================================
# 1. CONFIGURAÇÕES
# ==========================================
# Caminho para os resultados resilientes
RESULTS_DIR_RESILIENT = '../../results/results_resilient/*'

# Ajuste este valor para o número de rodadas (seeds) que você executou.
# Isso serve para calcular a média de SFCs afetadas por simulação.
N_SEEDS = 18


NOME_ALGORITMOS_MAP = {
    "DRLMASKABLEPPO": "REINFORCE",
    "REPLICMASKABLEPPO": "RELIC",
    "drl": "Deep RL",
    "random": "Random Placement"
}


# Definição de Cores (Gradiente para 3 estados por Nível de Risco)
# Formato: 'Risco': (Cor Falha, Cor Slow Recover, Cor Fast Recover)
# High: Vermelhos | Medium: Laranjas | Low: Verdes
CORES_ESTADOS = {
    'High':   ('#d62728', '#ff9896', '#ffcccc'), # Vermelho (Forte, Médio, Claro)
    'Medium': ('#ff7f0e', '#ffbb78', '#ffdfbf'), # Laranja
    'Low':    ('#2ca02c', '#98df8a', '#cceec4')  # Verde
}

# Hachuras para ajudar a diferenciar Status (visual acessível e P&B friendly)
HACHURAS = {
    'Failed': '///',        # Listrado forte para falha
    'Slow Recover': '...',  # Pontilhado para recuperação lenta (reinstanciação)
    'Fast Recover': ''      # Liso para recuperação rápida (backup)
}

# ==========================================
# 2. CARREGAMENTO DOS DADOS
# ==========================================
def carregar_dados_resiliencia_v3(diretorio_pattern):
    data = []
    pastas = glob.glob(diretorio_pattern)
    print(f"Lendo {len(pastas)} pastas de resultados...")
    
    for pasta in pastas:
        # Extrai nome do algoritmo da pasta
        dir_name = os.path.basename(os.path.normpath(pasta))
        raw_name = dir_name.split('_')[0]
        # Aplica o mapa de nomes se você tiver definido anteriormente (NOME_ALGORITMOS_MAP)
        # nome_alg = NOME_ALGORITMOS_MAP.get(raw_name, raw_name)
        nome_alg = NOME_ALGORITMOS_MAP.get(raw_name, raw_name) # Fallback se não tiver mapa
        
        files = glob.glob(os.path.join(pasta, "resilient_results_*.csv"))
        for f in files:
            try:
                df = pd.read_csv(f)
                
                # Verifica se as colunas novas existem antes de processar
                if 'risk_level' in df.columns and 'final_status' in df.columns:
                    # Filtra apenas o necessário
                    temp_df = df[['risk_level', 'final_status']].copy()
                    temp_df['algorithm'] = nome_alg
                    data.append(temp_df)
            except Exception as e:
                pass
                
    if not data:
        return pd.DataFrame()
        
    return pd.concat(data, ignore_index=True)

# ==========================================
# 3. PROCESSAMENTO E PLOTAGEM
# ==========================================

print("Iniciando processamento dos novos dados de resiliência...")
df_resil = carregar_dados_resiliencia_v3(RESULTS_DIR_RESILIENT)

if not df_resil.empty:
    # Agrupa por Algoritmo, Risco e Status e conta as ocorrências
    df_agg = df_resil.groupby(['algorithm', 'risk_level', 'final_status']).size().reset_index(name='total_count')
    
    # Normaliza para obter a média por simulação (dividir pelo numero de seeds)
    df_agg['avg_count'] = df_agg['total_count'] / N_SEEDS

    # Ordenação dos Algoritmos (Opcional: defina sua ordem de preferência aqui)
    algs = sorted(df_agg['algorithm'].unique())
    indices = np.arange(len(algs))
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    width = 0.65
    bottoms = np.zeros(len(algs))
    
    # Ordem de empilhamento visual (De baixo para cima):
    # 1. Low Risk (Fast -> Slow -> Fail)
    # 2. Medium Risk (Fast -> Slow -> Fail)
    # 3. High Risk (Fast -> Slow -> Fail)
    ordem_plot = [
        ('Low', 'Fast Recover'), ('Low', 'Slow Recover'), ('Low', 'Failed'),
        ('Medium', 'Fast Recover'), ('Medium', 'Slow Recover'), ('Medium', 'Failed'),
        ('High', 'Fast Recover'), ('High', 'Slow Recover'), ('High', 'Failed')
    ]
    
    legend_handles = []
    processed_labels = set()

    for risco, status in ordem_plot:
        valores = []
        for alg in algs:
            # Busca o valor correspondente. Se não existir, é 0.
            filtro = (df_agg['algorithm'] == alg) & \
                     (df_agg['risk_level'] == risco) & \
                     (df_agg['final_status'] == status)
            
            row = df_agg[filtro]
            val = row['avg_count'].values[0] if not row.empty else 0
            valores.append(val)
        
        valores = np.array(valores)
        
        # Definição de Estilo
        cores_risco = CORES_ESTADOS.get(risco, ('gray', 'gray', 'gray'))
        if status == 'Failed':
            cor = cores_risco[0]
        elif status == 'Slow Recover':
            cor = cores_risco[1]
        else: # Fast
            cor = cores_risco[2]
            
        hatch = HACHURAS.get(status, '')
        label_text = f"{risco} - {status}"
        
        # Plot da Barra
        ax.bar(indices, valores, width, bottom=bottoms, color=cor, 
               edgecolor='black', linewidth=0.5, hatch=hatch, alpha=0.9)
        
        # Atualiza o fundo para a próxima camada
        bottoms += valores
        
        # Cria handle para legenda manual (para garantir ordem correta)
        if label_text not in processed_labels:
            legend_handles.append(Patch(facecolor=cor, edgecolor='black', hatch=hatch, label=label_text))
            processed_labels.add(label_text)

    # Estilização do Eixo e Títulos
    ax.set_ylabel('Avg. Affected SFCs per Simulation', fontsize=16)
    ax.set_title('Resilience Profile: Impact by Risk & Recovery Type', fontsize=18, pad=20, fontweight='bold')
    
    ax.set_xticks(indices)
    ax.set_xticklabels(algs, fontsize=14, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Legenda Externa (Invertida para coincidir visualmente com a ordem das barras)
    ax.legend(handles=legend_handles[::-1], loc='center left', bbox_to_anchor=(1.02, 0.5), 
              fontsize=12, frameon=False, title="Risk & Outcome")
    
    plt.tight_layout()
    plt.savefig('detailed_resilience_new_cols.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
else:
    print("Nenhum dado encontrado com as novas colunas ('risk_level', 'final_status').")
    print("Certifique-se de rodar a simulação novamente com o código atualizado.")

# %%
# %% [Célula Nova] Gráfico de Resiliência Normalizado (100% Stacked Bar Chart)

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
from matplotlib.patches import Patch

# ==========================================
# 1. CONFIGURAÇÕES VISUAIS (Idênticas ao seu gráfico original)
# ==========================================
RESULTS_DIR_RESILIENT = '../../results/results_resilient/*'
N_SEEDS = 12  # Ajuste conforme o número de seeds que rodou

NOME_ALGORITMOS_MAP = {
    "DRLMASKABLEPPO": "REINFORCE",
    "REPLICMASKABLEPPO": "RELIC",
    "drl": "Deep RL",
    "random": "Random Placement"
}

# Cores: (Failed/Dark, Slow/Medium, Fast/Light)
CORES_ESTADOS = {
    'High':   ('#d62728', '#ff9896', '#ffcccc'), # Vermelhos
    'Medium': ('#ff7f0e', '#ffbb78', '#ffdfbf'), # Laranjas
    'Low':    ('#2ca02c', '#98df8a', '#cceec4')  # Verdes
}

# Hachuras para diferenciar o tipo de recuperação
HACHURAS = {
    'Failed': '///',        
    'Slow Recover': '...',  
    'Fast Recover': ''      
}

# ==========================================
# 2. CARREGAMENTO E PROCESSAMENTO
# ==========================================
def gerar_grafico_normalizado(diretorio_pattern):
    data = []
    pastas = glob.glob(diretorio_pattern)
    print(f"Lendo {len(pastas)} pastas...")
    
    # 1. Ler todos os CSVs
    for pasta in pastas:
        dir_name = os.path.basename(os.path.normpath(pasta))
        raw_name = dir_name.split('_')[0]
        nome_alg = NOME_ALGORITMOS_MAP.get(raw_name, raw_name)
        
        files = glob.glob(os.path.join(pasta, "resilient_results_*.csv"))
        for f in files:
            try:
                df = pd.read_csv(f)
                if 'risk_level' in df.columns and 'final_status' in df.columns:
                    temp_df = df[['risk_level', 'final_status']].copy()
                    temp_df['algorithm'] = nome_alg
                    data.append(temp_df)
            except: pass
                
    if not data:
        print("Nenhum dado encontrado.")
        return

    df_resil = pd.concat(data, ignore_index=True)

    # 2. Agrupar os dados
    # Conta quantos casos houve de cada tipo (Ex: RELIC -> High -> Fast Recover = 50)
    df_agg = df_resil.groupby(['algorithm', 'risk_level', 'final_status']).size().reset_index(name='count')
    
    # Divide pelo número de seeds para ter a média por simulação (como no gráfico original)
    df_agg['avg_count'] = df_agg['count'] / N_SEEDS

    # 3. NORMALIZAÇÃO (O "Pulo do Gato")
    # Calcula o total de falhas PARA CADA ALGORITMO
    total_por_algoritmo = df_agg.groupby('algorithm')['avg_count'].transform('sum')
    
    # Cria a coluna de porcentagem: (Valor / Total do Algoritmo) * 100
    df_agg['percentage'] = (df_agg['avg_count'] / total_por_algoritmo) * 100

    # ==========================================
    # 4. PLOTAGEM
    # ==========================================
    algs = sorted(df_agg['algorithm'].unique())
    indices = np.arange(len(algs))
    
    fig, ax = plt.subplots(figsize=(10, 8)) # Formato um pouco mais quadrado/alto
    width = 0.6
    bottoms = np.zeros(len(algs))
    
    # Ordem exata de empilhamento (de baixo para cima)
    ordem_plot = [
        ('Low', 'Fast Recover'), ('Low', 'Slow Recover'), ('Low', 'Failed'),
        ('Medium', 'Fast Recover'), ('Medium', 'Slow Recover'), ('Medium', 'Failed'),
        ('High', 'Fast Recover'), ('High', 'Slow Recover'), ('High', 'Failed')
    ]
    
    legend_handles = []
    processed_labels = set()

    for risco, status in ordem_plot:
        valores_percentuais = []
        
        for alg in algs:
            # Filtra o valor específico para esse algoritmo e combinação
            row = df_agg[(df_agg['algorithm'] == alg) & 
                         (df_agg['risk_level'] == risco) & 
                         (df_agg['final_status'] == status)]
            
            pct = row['percentage'].values[0] if not row.empty else 0
            valores_percentuais.append(pct)
        
        # Definição de Cores e Estilos
        cores_risco = CORES_ESTADOS.get(risco)
        if status == 'Failed': cor = cores_risco[0]
        elif status == 'Slow Recover': cor = cores_risco[1]
        else: cor = cores_risco[2]
            
        hatch = HACHURAS.get(status, '')
        
        # Plota a barra empilhada (usando percentage em vez de avg_count)
        ax.bar(indices, valores_percentuais, width, bottom=bottoms, color=cor, 
               edgecolor='black', linewidth=0.5, hatch=hatch, alpha=0.9)
        
        bottoms += np.array(valores_percentuais)
        
        # Cria legenda
        label_text = f"{risco} - {status}"
        if label_text not in processed_labels:
            legend_handles.append(Patch(facecolor=cor, edgecolor='black', hatch=hatch, label=label_text))
            processed_labels.add(label_text)

    # ==========================================
    # 5. AJUSTES FINAIS
    # ==========================================
    ax.set_ylabel('Percentage of Affected SFCs (%)', fontsize=18)  # Eixo Y agora é %
    ax.set_title('Normalized Impact Profile (100% Stacked)', fontsize=18, fontweight='bold', pad=20)
    
    ax.set_ylim(0, 100) # Garante que vai de 0 a 100
    ax.set_xticks(indices)
    ax.set_xticklabels(algs, fontsize=16)
    
    # Linha pontilhada no 50% para ajudar a ver a proporção
    ax.axhline(50, color='gray', linestyle='--', alpha=0.3) 

    # Legenda fora do gráfico
    ax.legend(handles=legend_handles[::-1], title="Risk & Outcome", 
              loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=12, frameon=False)
    
    plt.tight_layout()
    plt.savefig('resilience_profile_normalized_100.pdf', dpi=300, bbox_inches='tight')
    plt.show()

# Executar
gerar_grafico_normalizado(RESULTS_DIR_RESILIENT)



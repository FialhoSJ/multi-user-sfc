from pathlib import Path

# Garante a existência de logs/ (usada pelos FileHandlers dos algoritmos)
# antes de qualquer import de muar_sfc.* durante a coleta dos testes.
ROOT_DIR = Path(__file__).resolve().parent.parent
(ROOT_DIR / "logs").mkdir(exist_ok=True)

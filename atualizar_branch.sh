#!/bin/bash

# Vai para a branch net_updates
git checkout net_updates || { echo "Erro ao mudar para a branch net_updates"; exit 1; }

# Busca as atualizações do repositório remoto
git fetch origin || { echo "Erro ao buscar atualizações do repositório remoto"; exit 1; }

# Atualiza a branch net_updates com as mudanças do origin/net_updates
git merge origin/net_updates || { echo "Erro ao fazer merge com origin/net_updates"; exit 1; }

# Vai para a branch net_updates_david
git checkout net_updates_david || { echo "Erro ao mudar para a branch net_updates_david"; exit 1; }

# Mescla as atualizações da net_updates na sua branch
git merge net_updates || { echo "Erro ao fazer merge com net_updates"; exit 1; }

echo "Branches atualizadas com sucesso!"
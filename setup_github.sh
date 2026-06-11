#!/bin/bash
# =============================================================
# setup_github.sh
# Inicializa o repositório local e faz push para o GitHub.
# Execute este script UMA VEZ após clonar / criar o projeto.
# =============================================================

set -e

REMOTE="https://github.com/marcelo-f-rodrigues/trading_backtest.git"

echo "=== Trading Backtest Framework — Setup GitHub ==="

# 1. Inicializar git (se ainda não inicializado)
if [ ! -d ".git" ]; then
    git init
    echo "[OK] Repositório git inicializado."
fi

# 2. Configurar remote
if git remote get-url origin &>/dev/null; then
    echo "[INFO] Remote 'origin' já existe: $(git remote get-url origin)"
else
    git remote add origin "$REMOTE"
    echo "[OK] Remote adicionado: $REMOTE"
fi

# 3. Criar pastas que precisam existir no repo (mas são ignoradas pelo .gitignore)
mkdir -p data/raw data/processed results/reports results/charts results/rankings

# Adicionar .gitkeep para garantir que as pastas existam no repo
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch results/reports/.gitkeep
touch results/charts/.gitkeep
touch results/rankings/.gitkeep

# 4. Adicionar tudo
git add .
git commit -m "feat: estrutura inicial do framework de backtest"

# 5. Push
git branch -M main
git push -u origin main

echo ""
echo "=== Push concluído! ==="
echo "Acesse: $REMOTE"

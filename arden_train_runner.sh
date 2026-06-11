#!/bin/bash
# ARDEN 1.0 — Training Runner Script
# Copyright 2026 Nex Bridge Solutions LLC — David Ernesto Arriaga Pineda
# SPDX-License-Identifier: Arden Community License v1.0
#
# Instalar como servicio:
#   sudo cp arden_train_runner.sh /opt/arden/
#   sudo chmod +x /opt/arden/arden_train_runner.sh
#   sudo cp arden-train.service /etc/systemd/system/
#   sudo systemctl daemon-reload
#   sudo systemctl start arden-train
#   sudo systemctl enable arden-train

set -euo pipefail

VENV="/opt/arden/venv"
WORKDIR="/opt/arden"
LOG_DIR="/opt/arden/logs"
LOG_FILE="$LOG_DIR/arden_training.log"

mkdir -p "$LOG_DIR"

echo "============================================"
echo "  ARDEN 1.0 — Training Service"
echo "  Inicio: $(date)"
echo "  PID: $$"
echo "============================================"

source "$VENV/bin/activate"

python3 -c "import torch; print(f'PyTorch {torch.__version__} — device: GPU')"

cd "$WORKDIR"
python3 -u train.py

EXIT_CODE=$?

echo ""
echo "============================================"
echo "  Fin: $(date)"
echo "  Exit code: $EXIT_CODE"
echo "============================================"

exit $EXIT_CODE
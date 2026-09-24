#!/bin/bash
# Model C (from-scratch skip-gram) on text8, on one ICE GPU. Submit from the project root:
#   sbatch slurm/c_text8.sbatch
# Checkpoints after every epoch to models/C_text8.ckpt; resubmitting resumes from there.
#SBATCH -J c-text8
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o slurm/c_text8-%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv-ice/bin/activate
gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-')
export BDS_MACHINE="pace-ice-$gpu"
echo "node $(hostname)  gpu $gpu  start $(date)"
nvidia-smi --query-gpu=name,driver_version,ecc.mode.current,ecc.errors.uncorrected.volatile.total --format=csv
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; x = torch.ones(1, device='cuda'); print('cuda ok', torch.cuda.get_device_name(0))"
python -u -m src.run_e --text8 --device cuda
echo "done $(date)"

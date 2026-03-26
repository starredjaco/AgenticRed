#!/bin/bash
#SBATCH -J eval
#SBATCH -p spyder
#SBATCH -t 7-00:00 
#SBATCH --mem=10G
#SBATCH -o slurm/hostname_%j.out
#SBATCH -e slurm/hostname_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REDTEAM_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="./configs/eval_easyjailbreak.yaml"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: bash scripts/eval.sh --config <yaml>"
            exit 1
            ;;
    esac
done

if [[ -z "${CONFIG_PATH}" ]]; then
    echo "Missing required argument: --config <yaml>"
    exit 1
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config file not found: ${CONFIG_PATH}"
    exit 1
fi

python -u search.py \
    --mode evaluate \
    --config "${CONFIG_PATH}"
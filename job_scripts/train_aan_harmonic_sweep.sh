#!/bin/bash
#PBS -q short-g
#PBS -l select=1
#PBS -l walltime=04:00:00
#PBS -N my_acion_angle
#PBS -o logs/
#PBS -e logs/
#PBS -j oe
#PBS -W group_list=gj26

module purge
module load singularity

# --- ログは $PBS_O_WORKDIR に出る ---
cd $PBS_O_WORKDIR

export CUDNN_LIB_DIR=/work/gj26/b20109/.local/lib/python3.12/site-packages/nvidia/cudnn/lib
source /work/gj26/b20109/env_cudnn98.sh

export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export WANDB_API_KEY=ac9bc3f259163957d95686abca5fb49df1713b65

TODAY=$(date '+%Y%m%d')

# --- Sweep IDを環境変数またはファイルから取得 ---
if [ -z "$WANDB_SWEEP_ID" ]; then
    if [ -f "sweep_id.txt" ]; then
        WANDB_SWEEP_ID=$(cat sweep_id.txt)
    else
        echo "ERROR: WANDB_SWEEP_ID not set and sweep_id.txt not found"
        echo "Run: wandb sweep configs/_sweep.yaml"
        echo "Then: echo 'your_sweep_id' > sweep_id.txt"
        exit 1
    fi
fi

echo "Job started at: $(date)"

# --- Single training run ---
singularity exec --nv \
  --bind $(pwd):/workspace \
  --bind /etc/pki/tls/certs/ca-bundle.crt:/etc/pki/tls/certs/ca-bundle.crt \
  ~/singularity/pytorch_25.01.sif \
  python -m wandb agent "$WANDB_SWEEP_ID"


STATUS=$?   # 0=正常, それ以外=異常

# ---- Slack 通知 ----

JOB_NAME=$PBS_JOB_NAME
JOB_ID=$PBS_JOBID
NODE_NAME=$(hostname)

send_slack() {         # 小さなヘルパー関数
  curl -s -X POST -H 'Content-type: application/json' \
       --data "{\"text\":\"$1\"}" "$SLACK_WEBHOOK"
}

if [ "$STATUS" -eq 0 ]; then
    MESSAGE="✅ *Sweep Job Finished Successfully*\n> Job Name: \`$JOB_NAME\`\n> Job ID: \`$JOB_ID\`\n> Node: \`$NODE_NAME\`\n> Sweep ID: \`$WANDB_SWEEP_ID\`"
    send_slack "$MESSAGE"
else
    MESSAGE="❌ *Sweep Job Failed*\n> Job Name: \`$JOB_NAME\`\n> Job ID: \`$JOB_ID\`\n> Node: \`$NODE_NAME\`\n> Sweep ID: \`$WANDB_SWEEP_ID\`\n> Exit Code: \`$STATUS\`"
    send_slack "$MESSAGE"
fi
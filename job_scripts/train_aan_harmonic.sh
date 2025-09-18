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

export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
# export WANDB_API_KEY=ac9bc3f259163957d95686abca5fb49df1713b65
# export WANDB_PROJECT=my_acion_angle

TODAY=$(date '+%Y%m%d')

# --- Sweep Agent実行 ---
singularity exec --nv \
  --bind $(pwd):/workspace \
  ~/singularity/pytorch_25.01.sif \
  python scripts/e2025_0918_aan_harmonic_motion.py \
    --wandb-project aan_harmonic \
    --experiment_name aan_harmonic \
    --num_steps 5000 \
    --test-time-jumps [1, 2, 5, 10, 20, 50] \
    --alpha 1.0 \
    --normalize-qp False \
    --dim-hidden-list [32, 32, 32] \
    --num-gsblocks 20 \
    --activation sigmoid \
    --mlp-res-connection True \
    --batch-size 100 \
    --learning-rate 0.001 \
    --weight-decay 0.0001 \
    --seed 42 \
    --log-every 100 \
    --save-every 1000 \
    --save-dir results/${TODAY}_aan_harmonic
    



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
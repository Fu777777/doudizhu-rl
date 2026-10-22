#!/usr/bin/env bash
# 用修复后的 evaluate.py（随机农民每局重播种）对 3 个种子 × 6 种地主策略各评估 1000 局
cd "$(dirname "$0")"; source .venv/bin/activate; mkdir -p logs_seeds/eval_fixed
M="dqn double_dqn dueling_double bc bc_double bc_double_aux"
for s in 1 2; do M="$M dqn_s$s double_s$s dueling_double_s$s bc_s$s bc_double_s$s bc_double_aux_s$s"; done
for m in $M; do echo $m; done | xargs -P 7 -I{} sh -c 'python evaluate.py --games 1000 --models checkpoints/{}.pt > logs_seeds/eval_fixed/{}.txt 2>&1'
echo EVAL_DONE

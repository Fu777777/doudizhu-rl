#!/usr/bin/env bash
# 多种子实验：seed 1、2（seed 0 为原始实验），另对 seed 0 的 Double DQN 复跑一次检验可复现性
cd "$(dirname "$0")"
source .venv/bin/activate
mkdir -p logs_seeds
run() { local log=$1; shift; local t0=$SECONDS; python "$@" > logs_seeds/$log 2>&1; echo "WALL $((SECONDS-t0)) s | cmd: python $*" >> logs_seeds/$log; }
# 先做 BC（专家数据固定种子 500000，各种子数据相同，只有网络初始化/打乱顺序不同）
for s in 1 2; do run bc_s$s.txt imitation.py --games 3000 --epochs 10 --seed $s --name bc_s$s & done
wait
run double_s0_rerun.txt train.py --algo double --seed 0 --name rerun_double_s0 &
for s in 1 2; do
  run dqn_s$s.txt train.py --algo dqn --seed $s --name dqn_s$s &
  run double_s$s.txt train.py --algo double --seed $s --name double_s$s &
  run dueling_s$s.txt train.py --algo double --dueling --seed $s --name dueling_double_s$s &
done
wait
for s in 1 2; do
  run bc_double_s$s.txt train.py --algo double --init checkpoints/bc_s$s.pt --eps-start 0.2 --seed $s --name bc_double_s$s &
  run bc_double_aux_s$s.txt train.py --algo double --init checkpoints/bc_s$s.pt --eps-start 0.2 --bc-weight 1.0 --seed $s --name bc_double_aux_s$s &
done
wait
echo ALL_DONE

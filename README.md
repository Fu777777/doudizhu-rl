# 斗地主强化学习 AI（DQN / Double DQN，简化规则）

用 PyTorch 从零实现 DQN 与 Double DQN，在自定义 gymnasium 斗地主环境中训练“地主”智能体，对手为随机农民和规则农民。代码量约 400 行，一晚上能读完。

## 1. 思路

- **问题建模**：智能体固定当地主（20 张牌，先出），两个农民由固定策略控制并被封装进环境内部。对智能体来说这就是一个单智能体 MDP：每次 `step` = 我出一手 + 两个农民各出一手。
- **奖励**：稀疏奖励，地主赢 +1，输 -1，其余 0。
- **动作空间**：把所有可能的出牌枚举成固定的 274 个离散动作（过、单、对、三、三带一、顺子、炸弹、王炸）。每步由环境给出**合法动作掩码** `info["action_mask"]`，选择动作时把非法动作的 Q 值置为 -inf。
- **状态编码（81 维，均归一化到 [0,1]）**：自己手牌计数(15) + 全场已出牌(15) + 需要压的上一手牌(15) + 三家剩余牌数(3) + 上一手是谁出的(3，相对位置 one-hot) + 两个农民各自已出牌(15×2)。
- **算法**：MLP Q 网络（81→256→256→274），经验回放、目标网络（每 500 次更新硬同步）、epsilon 从 1.0 线性衰减到 0.05、Huber 损失、梯度裁剪。Double DQN 用在线网络选下一步动作、目标网络估值，缓解过估计。计算目标值时下一状态的 max 也只在其**合法动作**中取。
- **课程式对手**：训练时按局随机选择对手，规则农民比例从 20% 逐步升到 80%。

### 简化规则
无叫地主（玩家 0 固定当地主，拿底牌）；牌型只保留 单/对/三/三带一/顺子(5~12 张)/炸弹/王炸，**不含**连对、飞机、三带二、四带二；不计倍数。

## 2. 目录结构

```
doudizhu-rl/
├── doudizhu/
│   ├── cards.py     # 牌点、动作枚举(274个)、压牌规则、合法动作掩码(向量化)
│   ├── env.py       # gymnasium 环境 DoudizhuEnv：发牌、状态编码、step
│   ├── agents.py    # RandomAgent 随机策略、RuleAgent 简单规则策略
│   └── dqn.py       # QNet、ReplayBuffer、DQNAgent（DQN / Double DQN）
├── train.py         # 训练脚本
├── evaluate.py      # 评估脚本：各种地主 vs 规则/随机农民 胜率
├── logs_*.txt       # 本机实际运行日志
└── requirements.txt
```

## 3. 如何运行（Windows）

建议把项目放在 `D:\Projects\doudizhu-rl`，在 PowerShell 中：

```powershell
cd D:\Projects\doudizhu-rl
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install gymnasium numpy

python train.py --algo dqn    --episodes 3000   # 约 3 分钟(CPU)，输出 checkpoints\dqn.pt
python train.py --algo double --episodes 3000   # 输出 checkpoints\double_dqn.pt
python evaluate.py --games 1000
```

（Linux/macOS 激活虚拟环境用 `source .venv/bin/activate`。若 PowerShell 禁止脚本，先执行 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`。）

## 4. 实际运行结果

环境：Linux 容器、CPU 单线程、Python 3.13、torch 2.14.1+cpu、gymnasium 1.4.0。每种算法训练 3000 局（种子 0），各约 3.3 分钟。完整日志见 `logs_dqn.txt`、`logs_double.txt`、`logs_eval.txt`。

训练过程（每 500 局用贪心策略对规则农民评估 200 局）：

| 局数 | DQN 评估胜率 | Double DQN 评估胜率 |
|---|---|---|
| 500 | 2.0% | 3.5% |
| 1000 | 2.5% | 7.5% |
| 1500 | 4.5% | 7.0% |
| 2000 | 4.0% | 6.0% |
| 2500 | 7.0% | 7.5% |
| 3000 | 4.5% | 7.5% |

最终评估（`python evaluate.py --games 1000`，所有策略使用同一组 1000 副牌）：

| 地主策略 | vs 规则农民 | vs 随机农民 |
|---|---|---|
| 随机地主（基线） | 1.8% | 36.4% |
| 规则地主（基线） | 58.9% | 93.8% |
| DQN | 4.7% | 52.2% |
| Double DQN | 8.1% | 59.5% |

**结论（如实）**：几分钟的训练后，DQN / Double DQN 明显好于随机策略（对随机农民 36.4% → 52~60%，对规则农民 1.8% → 4.7~8.1%），Double DQN 在本次运行中略好于 DQN；但距离手写规则地主（58.9%）还差很远。这只是单个种子的短训练结果，不足以得出统计意义上的严格结论。

## 5. 局限与改进方向

- 274 维输出头把动作当作互相独立的类别，无法泛化“相似牌型”。DouZero 的做法是把**动作也编码成特征**输入网络，输出单个 Q(s,a)，效果好很多。
- 奖励只在终局给出且回合较长，DQN 信用分配困难；可改用蒙特卡洛回报（DouZero 即用 Deep Monte-Carlo）、n-step 回报或优先经验回放。
- 训练后期胜率与 loss 都有波动，说明随着对手比例变化和自举误差，Q 值不够稳定；可调小学习率、软更新目标网络、增加训练局数。
- 状态中加入出牌历史序列（LSTM 编码）、补全牌型、加入叫地主与农民智能体自博弈。

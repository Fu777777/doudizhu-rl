# 斗地主强化学习 AI（DQN / Double DQN / Dueling DQN + 模仿学习预训练，简化规则）

用 PyTorch 从零实现 DQN、Double DQN、Dueling DQN，并加入模仿学习（行为克隆）预训练，在自定义 gymnasium 斗地主环境中训练“地主”智能体，对手为随机农民和规则农民。代码量约 550 行，一晚上能读完。

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
├── imitation.py     # 模仿学习：RuleAgent 自对弈生成专家数据 + 行为克隆(BC)预训练
├── train.py         # 训练脚本（--dueling / --init 预训练初始化 / --bc-weight 辅助损失）
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
python train.py --algo double --dueling --name dueling_double          # Dueling Double DQN
python imitation.py --games 3000 --epochs 10                               # 约 15 秒，输出 checkpoints\bc.pt、expert.pt
python train.py --algo double --init checkpoints\bc.pt --eps-start 0.2 --name bc_double
python train.py --algo double --init checkpoints\bc.pt --eps-start 0.2 --bc-weight 1.0 --name bc_double_aux
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

## 4.5 新增：模仿学习预训练 + Dueling DQN

### 设计
- **专家数据**（`imitation.py`）：RuleAgent 当地主、对两个 RuleAgent 农民打 3000 局（种子区间 500000+，与训练/评估不重叠），记录地主每一步的 (状态, 动作, 合法掩码)，共 44784 个样本，专家本身胜率 60.9%。
- **行为克隆**：直接把 Q 网络的 274 维输出当作 logits，非法动作置 -1e9 后做交叉熵，10 个 epoch。训练完把权重同时拷给目标网络。
- **DQN 微调**：`train.py --init checkpoints/bc.pt`，探索率从 0.2 起（从 1.0 起会立刻把预训练策略打乱）。
- **可选 DQfD 式辅助损失**：`--bc-weight 1.0` 时，每次 TD 更新再从专家数据里采一个 batch，加一项掩码交叉熵。
- **Dueling**（`--dueling`）：共享两层 MLP 后分成 V(s) 和 A(s,a) 两个头，Q = V + A − mean(A)。

### 实际结果（种子 0，每个 DQN 变体 3000 局，各约 3.5~4 分钟；日志 `logs_bc.txt`、`logs_dueling.txt`、`logs_bc_double*.txt`）

BC 预训练：loss 0.2714 → 0.0131，训练集动作与专家一致率 94.1% → 99.6%，用时 12 s；对规则农民 200 局评估 60.0%。

训练过程（每 500 局贪心评估，对规则农民 200 局）：

| 局数 | DQN | Double | Dueling Double | BC→Double 微调 | BC→Double 微调+BC辅助损失 |
|---|---|---|---|---|---|
| 500 | 2.0% | 3.5% | 9.5% | 1.0% | 59.0% |
| 1000 | 2.5% | 7.5% | 4.5% | 3.0% | 58.0% |
| 1500 | 4.5% | 7.0% | 10.0% | 14.5% | 55.0% |
| 2000 | 4.0% | 6.0% | 10.0% | 6.5% | 57.0% |
| 2500 | 7.0% | 7.5% | 8.5% | 12.5% | 58.5% |
| 3000 | 4.5% | 7.5% | 4.5% | 17.0% | 56.0% |

最终评估（`python evaluate.py --games 1000`，同一组 1000 副牌，日志 `logs_eval.txt`）：

| 地主策略 | vs 规则农民 | vs 随机农民 |
|---|---|---|
| 随机地主（基线） | 1.8% | 36.4% |
| 规则地主（基线，即专家） | 58.9% | 93.8% |
| DQN | 4.7% | 52.2% |
| Double DQN | 8.1% | 59.5% |
| Dueling Double DQN | 7.0% | 56.9% |
| 仅 BC（行为克隆） | 58.6% | **95.4%** |
| BC 初始化 → Double DQN 微调 | 14.2% | 67.4% |
| BC 初始化 → Double DQN 微调 + BC 辅助损失 | 56.0% | 84.7% |

### 结论（如实）
1. **模仿学习是本项目中提升最大的手段**：纯 BC 几乎复刻专家（对规则农民 58.6% vs 专家 58.9%），对随机农民甚至略高于专家（95.4% vs 93.8%，1000 局的差异在统计噪声范围内）。
2. **直接用 BC 权重初始化再做 DQN 微调会“灾难性遗忘”**：第 500 局评估就从 60% 掉到 1.0%，最终 14.2%。原因：BC 输出是 logits，量纲与 Q 值（±1 回报）不一致，最初几百次 TD 更新会把整张输出表重新拉到 Q 值尺度，策略随之被打乱。不过它最终仍好于从零训练的 DQN（14.2% vs 8.1%），说明预训练的表示仍有帮助。
3. **加 BC 辅助损失（DQfD 思路）能保住专家水平**（56.0%），但 3000 局内没有超越专家；对随机农民反而从 95.4% 降到 84.7%。即：短训练下 RL 微调没有带来正收益。
4. **Dueling 在本次单种子实验中无明显收益**（7.0% vs Double 8.1%），差异小于评估噪声。
5. 所有数字都是单种子、短训练，只能说明趋势，不构成统计显著结论。

## 5. 局限与改进方向

- 预训练 → 微调：更稳妥的做法是先只用专家数据“预热”Q 头（如 DQfD 的 large-margin 损失让专家动作 Q 值高出其他动作一个间隔），或用策略梯度类方法（BC 直接是策略初始化）微调。

- 274 维输出头把动作当作互相独立的类别，无法泛化“相似牌型”。DouZero 的做法是把**动作也编码成特征**输入网络，输出单个 Q(s,a)，效果好很多。
- 奖励只在终局给出且回合较长，DQN 信用分配困难；可改用蒙特卡洛回报（DouZero 即用 Deep Monte-Carlo）、n-step 回报或优先经验回放。
- 训练后期胜率与 loss 都有波动，说明随着对手比例变化和自举误差，Q 值不够稳定；可调小学习率、软更新目标网络、增加训练局数。
- 状态中加入出牌历史序列（LSTM 编码）、补全牌型、加入叫地主与农民智能体自博弈。

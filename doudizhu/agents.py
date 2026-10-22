"""对手策略：随机策略与简单规则策略。地主/农民都能用。"""
import numpy as np

from .cards import ACTIONS, PASS

# 自由出牌时，同等点数下优先出“带走牌多”的牌型
_LEAD_PRIORITY = {"straight": 0, "triple1": 1, "triple": 2, "pair": 3, "single": 4}


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)

    def act(self, view, mask):
        return int(self.rng.choice(np.flatnonzero(mask)))


class RuleAgent:
    """简单规则：
    1) 自由出牌：出点数最小的非炸弹牌，同点数优先顺子>三带一>三张>对子>单张；
    2) 跟牌：队友(农民之间)出的牌不压；否则用能压住的最小同类型牌；
    3) 炸弹/王炸只在对手剩牌 <=5 张时才用。"""

    def act(self, view, mask):
        legal = [i for i in np.flatnonzero(mask) if i != PASS]
        if not legal:
            return PASS
        me, lp = view["me"], view["last_player"]
        free = lp in (-1, me)
        if not free and me != 0 and lp != 0:  # 农民不压队友
            return PASS
        normal = [i for i in legal if ACTIONS[i].kind not in ("bomb", "rocket")]
        if free:
            return int(min(normal or legal, key=lambda i: (ACTIONS[i].rank, _LEAD_PRIORITY.get(ACTIONS[i].kind, 9), -ACTIONS[i].length)))
        if normal:
            return int(min(normal, key=lambda i: ACTIONS[i].rank))
        if view["left"][lp] <= 5:  # 对手快走完了才炸
            return int(min(legal, key=lambda i: ACTIONS[i].rank))
        return PASS

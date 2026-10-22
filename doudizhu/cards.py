"""牌面与动作定义：把所有可能的出牌枚举成一个固定大小的动作表。

牌点下标 0..14 依次为：3 4 5 6 7 8 9 10 J Q K A 2 小王 大王。
手牌用长度 15 的计数向量表示（每个点数有几张）。
"""
import numpy as np

RANKS = ["3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A", "2", "小王", "大王"]
NUM_RANKS = 15


def full_deck():
    """一副 54 张牌，返回点数下标列表。"""
    deck = []
    for r in range(13):
        deck += [r] * 4
    deck += [13, 14]
    return deck


class Action:
    """一个出牌动作。kind: pass/single/pair/triple/triple1/straight/bomb/rocket。
    rank 为主牌点数（顺子为起点），length 为顺子长度，counts 为所需牌的计数向量。"""

    def __init__(self, kind, rank=-1, length=0, counts=None):
        self.kind, self.rank, self.length = kind, rank, length
        self.counts = counts if counts is not None else np.zeros(NUM_RANKS, dtype=np.int8)

    def __repr__(self):
        if self.kind == "pass":
            return "过"
        cards = []
        for r in range(NUM_RANKS):
            cards += [RANKS[r]] * int(self.counts[r])
        return f"{self.kind}[{' '.join(cards)}]"


def _vec(d):
    v = np.zeros(NUM_RANKS, dtype=np.int8)
    for r, c in d.items():
        v[r] += c
    return v


def build_actions():
    """枚举全部动作（简化规则：不含连对、飞机、三带二、四带二）。"""
    acts = [Action("pass")]
    for r in range(15):
        acts.append(Action("single", r, 1, _vec({r: 1})))
    for r in range(13):
        acts.append(Action("pair", r, 2, _vec({r: 2})))
    for r in range(13):
        acts.append(Action("triple", r, 3, _vec({r: 3})))
    for r in range(13):  # 三带一，带的单牌不能与主牌同点
        for k in range(15):
            if k != r:
                acts.append(Action("triple1", r, 4, _vec({r: 3, k: 1})))
    for start in range(12):  # 顺子：3..A 中连续 >=5 张，不能含 2 和王
        for length in range(5, 13):
            if start + length <= 12:
                acts.append(Action("straight", start, length, _vec({r: 1 for r in range(start, start + length)})))
    for r in range(13):
        acts.append(Action("bomb", r, 4, _vec({r: 4})))
    acts.append(Action("rocket", 14, 2, _vec({13: 1, 14: 1})))
    return acts


ACTIONS = build_actions()
NUM_ACTIONS = len(ACTIONS)
PASS = 0


def can_beat(a, last):
    """动作 a 能否压过上家动作 last（last 为 None 表示自己自由出牌）。"""
    if a.kind == "pass":
        return last is not None  # 首家不能过
    if last is None:
        return True
    if last.kind == "rocket":
        return False
    if a.kind == "rocket":
        return True
    if a.kind == "bomb":
        return last.kind != "bomb" or a.rank > last.rank
    return a.kind == last.kind and a.length == last.length and a.rank > last.rank


# 预先算好的数组，用于向量化计算掩码（比逐个调用 can_beat 快很多）
_COUNTS = np.stack([a.counts for a in ACTIONS])            # [动作数, 15]
_KIND = np.array([a.kind for a in ACTIONS])
_RANK = np.array([a.rank for a in ACTIONS])
_LEN = np.array([a.length for a in ACTIONS])
_BOMBISH = (_KIND == "bomb") | (_KIND == "rocket")


def legal_mask(hand, last):
    """合法动作掩码：手里牌够且能压过上家（逻辑与 can_beat 一致，向量化实现）。"""
    have = np.all(_COUNTS <= hand, axis=1)
    if last is None:
        beat = _KIND != "pass"
    elif last.kind == "rocket":
        beat = _KIND == "pass"
    else:
        same = (_KIND == last.kind) & (_LEN == last.length) & (_RANK > last.rank)
        bomb = (_KIND == "bomb") & ((last.kind != "bomb") | (_RANK > last.rank))
        beat = same | bomb | (_KIND == "rocket") | (_KIND == "pass")
    return have & beat

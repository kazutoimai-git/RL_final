import numpy as np
import math

class MCTS():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}

    def getActionProb(self, canonicalBoard, temp=1):
        for _ in range(self.args['numMCTSSims']):
            self.search(canonicalBoard)

        s = self.game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa.get((s, a), 0) for a in range(self.game.getActionSize())]
        valids = self.game.getValidMoves(canonicalBoard, 1)
        
        # ✅ 修正済み: 無効な手の確率を0にするマスキング処理
        counts = np.array(counts) * valids

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts_sum = float(sum(counts))
        if counts_sum == 0:
            # 有効な手が一つもない場合（基本的には起こらないが安全策）
            return valids / np.sum(valids) if np.sum(valids) > 0 else valids

        probs = [c / counts_sum for c in counts]
        return probs

    def search(self, canonicalBoard):
        s = self.game.stringRepresentation(canonicalBoard)

        # ✅ 修正済み: ゲーム終了状態をチェック
        ended = self.game.getGameEnded(canonicalBoard, 1)
        if ended != 0:
            return -ended

        if s not in self.Ps:
            pi, v = self.nnet.predict(canonicalBoard)
            self.Ps[s] = pi
            self.Ns[s] = 0
            return -v

        valids = self.game.getValidMoves(canonicalBoard, 1)
        cur_best = -float('inf')
        best_act = -1

        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args['cpuct'] * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
                
                if u > cur_best:
                    cur_best = u
                    best_act = a
        
        a = best_act
        next_s, next_player = self.game.getNextState(canonicalBoard, 1, a)
        next_s = self.game.getCanonicalForm(next_s, next_player)

        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
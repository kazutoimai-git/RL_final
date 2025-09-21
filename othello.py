import numpy as np

class OthelloBoard:
    
    def getInitBoard(self):
        b = np.zeros((8, 8), dtype=int)
        b[3, 3] = -1; b[4, 4] = -1  # 白 (Player -1)
        b[3, 4] = 1;  b[4, 3] = 1   # 黒 (Player 1)
        return b

    def getActionSize(self):
        return 8 * 8

    def getNextState(self, board, player, action):
        if action == -1:  
            return (board, -player)

        r, c = action // 8, action % 8
        b = np.copy(board)
        
        if b[r, c] != 0:
            return (b, -player)

        b[r, c] = player
        flippable_stones = self._get_flippable_stones(b, r, c, player)
        for fr, fc in flippable_stones:
            b[fr, fc] = player
        return (b, -player)

    def getValidMoves(self, board, player):
        valids = [0] * self.getActionSize()
        for r in range(8):
            for c in range(8):
                if board[r, c] == 0:
                    if self._get_flippable_stones(board, r, c, player):
                        valids[r * 8 + c] = 1
        return np.array(valids)

    def getGameEnded(self, board, player):
        if np.sum(self.getValidMoves(board, player)) > 0:
            return 0
        if np.sum(self.getValidMoves(board, -player)) > 0:
            return 0
        
        black_stones = np.sum(board == 1)
        white_stones = np.sum(board == -1)
        if black_stones > white_stones:
            return 1 * player
        if white_stones > black_stones:
            return -1 * player
        return 1e-4

    def getCanonicalForm(self, board, player):
        return player * board

    def stringRepresentation(self, board):
        return board.tobytes()

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def _get_flippable_stones(self, board, r, c, player):
        flippable_stones = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            stones_in_direction = []
            row, col = r + dr, c + dc
            
            while 0 <= row < 8 and 0 <= col < 8 and board[row, col] == -player:
                stones_in_direction.append((row, col))
                row, col = row + dr, col + dc 
            
            if 0 <= row < 8 and 0 <= col < 8 and board[row, col] == player:
                flippable_stones.extend(stones_in_direction)
        return flippable_stones
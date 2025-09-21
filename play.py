import numpy as np
import os
from othello import OthelloBoard
from ai_model import OthelloNNet
from mcts import MCTS
import torch

# --- 設定 ---
MODEL_FILE = './temp/best.pth.tar'
NUM_MCTS_SIMS = 50 # AIの思考シミュレーション回数 (数値を上げると強くなる)

# --- NNetWrapperクラス (main.pyから抜粋・簡略化) ---
class NNetWrapper():
    def __init__(self, game):
        self.nnet = OthelloNNet(game)
        self.board_x, self.board_y = 8, 8
        self.action_size = game.getActionSize()
        # 対戦時は常に評価モード
        self.nnet.eval()

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64)).view(1, self.board_x, self.board_y)
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]

    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise Exception(f"モデルファイルが見つかりません: {filepath}")
        # CPUでモデルを読み込む
        map_location = 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

# --- 盤面表示用のヘルパー関数 ---
def display_board(board):
    print("  a b c d e f g h")
    for i in range(8):
        print(f"{i+1} ", end="")
        for j in range(8):
            stone = board[i, j]
            if stone == 1: print("B ", end="")      # 黒石
            elif stone == -1: print("W ", end="")   # 白石
            else: print(". ", end="")               # 空きマス
        print()

# --- ゲーム対戦のメイン関数 ---
def play_game():
    # --- 初期設定 ---
    game = OthelloBoard()
    nnet = NNetWrapper(game)
    
    print(f"AIモデル '{MODEL_FILE}' を読み込んでいます...")
    nnet.load_checkpoint(folder="", filename=MODEL_FILE)
    
    args = {'numMCTSSims': NUM_MCTS_SIMS, 'cpuct': 1.0}
    mcts = MCTS(game, nnet, args)

    # ✅ 修正済み: ターン管理のロジックを全面的に見直し
    board = game.getInitBoard()
    current_player = 1  # 1:黒(人間), -1:白(AI)

    # --- メインゲームループ ---
    while game.getGameEnded(board, current_player) == 0:
        print("\n" + "="*30)
        display_board(board)

        # 現在のプレイヤーが置ける場所を取得
        canonicalBoard = game.getCanonicalForm(board, current_player)
        valid_moves = game.getValidMoves(canonicalBoard, 1)

        # パスのチェック
        if np.sum(valid_moves) == 0:
            player_name = "あなた" if current_player == 1 else "AI"
            print(f"{player_name} はパスです。")
            current_player *= -1  # ターンを相手に渡す
            continue

        # プレイヤーに応じて手を選択
        if current_player == 1: # 人間のターン
            print("あなたの番です (黒: B)")
            valid_locations = [f"{(i//8)+1}{chr((i%8)+97)}" for i, v in enumerate(valid_moves) if v]
            print("置ける場所:", valid_locations)
            
            while True:
                move_str = input("どこに置きますか？ (例: 3d): ").lower()
                if len(move_str) == 2 and '1' <= move_str[0] <= '8' and 'a' <= move_str[1] <= 'h':
                    row, col = int(move_str[0]) - 1, ord(move_str[1]) - ord('a')
                    action = row * 8 + col
                    if valid_moves[action]:
                        break
                    else: print("その場所には置けません。")
                else: print("無効な入力です。'3d'のように入力してください。")
        
        else: # AIのターン
            print("AIの番です (白: W)... 考え中...")
            pi = mcts.getActionProb(canonicalBoard, temp=0)
            action = np.argmax(pi)

        # 選択した手を盤面に反映し、ターンを交代
        board, current_player = game.getNextState(board, current_player, action)

    # --- ゲーム終了 ---
    print("\n" + "="*30)
    print("--- ゲーム終了 ---")
    display_board(board)
    black, white = np.sum(board == 1), np.sum(board == -1)
    print(f"黒 (B): {black} 石, 白 (W): {white} 石")
    
    winner = game.getGameEnded(board, 1)
    if winner > 0: print("あなたの勝ちです！おめでとうございます！")
    elif winner < 0: print("AIの勝ちです。")
    else: print("引き分けです。")

if __name__ == "__main__":
    play_game()
import pygame
import sys
import os
import numpy as np
import time

from othello import OthelloBoard
from ai_model import OthelloNNet
from mcts import MCTS
import torch

# --- 定数と設定 ---
SCREEN_WIDTH, SCREEN_HEIGHT = 600, 700
BOARD_SIZE = 500
BOARD_X, BOARD_Y = (SCREEN_WIDTH - BOARD_SIZE) // 2, (SCREEN_HEIGHT - BOARD_SIZE) // 2 + 50
CELL_SIZE, STONE_RADIUS = BOARD_SIZE // 8, BOARD_SIZE // 16 - 5
BG_COLOR, BOARD_COLOR, BLACK, WHITE, GRID_COLOR = (20, 20, 20), (0, 100, 50), (10, 10, 10), (245, 245, 245), (30, 30, 30)
INFO_COLOR, BUTTON_COLOR, BUTTON_BORDER_COLOR, BUTTON_TEXT_COLOR = (200, 200, 200), (40, 40, 40), (80, 80, 80), (220, 220, 220)
STONE_SHADOW_COLOR = (0, 0, 0, 50)
MODEL_FILE, NUM_MCTS_SIMS = './temp/best.pth (2).tar', 1000

# --- Pygameの初期化 ---
# ※サーバー環境で実行する場合は、以下の2行のコメントを外します
# os.environ['SDL_VIDEODRIVER'] = 'dummy'
# os.environ['SDL_AUDIODRIVER'] = 'dummy'
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("強化学習オセロAI")
try:
    font = pygame.font.SysFont('helveticaneue', 40, bold=True)
    button_font = pygame.font.SysFont('helveticaneue', 30, bold=True)
except:
    font = pygame.font.SysFont('arial', 40, bold=True)
    button_font = pygame.font.SysFont('arial', 30, bold=True)

# (NNetWrapperクラスは変更なし)
class NNetWrapper():
    def __init__(self, game):
        self.nnet = OthelloNNet(game)
        self.board_x, self.board_y = 8, 8
        self.action_size = game.getActionSize()
        self.nnet.eval()
    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64)).view(1, self.board_x, self.board_y)
        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]
    def load_checkpoint(self, folder, filename):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath): raise Exception(f"モデルファイルが見つかりません: {filepath}")
        map_location = 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

# (draw_game, draw_menu関数は変更なし)
def draw_game(board, message):
    screen.fill(BG_COLOR)
    pygame.draw.rect(screen, GRID_COLOR, (BOARD_X - 4, BOARD_Y - 4, BOARD_SIZE + 8, BOARD_SIZE + 8))
    pygame.draw.rect(screen, BOARD_COLOR, (BOARD_X, BOARD_Y, BOARD_SIZE, BOARD_SIZE))
    for i in range(9):
        pygame.draw.line(screen, GRID_COLOR, (BOARD_X + i * CELL_SIZE, BOARD_Y), (BOARD_X + i * CELL_SIZE, BOARD_Y + BOARD_SIZE), 2)
        pygame.draw.line(screen, GRID_COLOR, (BOARD_X, BOARD_Y + i * CELL_SIZE), (BOARD_X + BOARD_SIZE, BOARD_Y + i * CELL_SIZE), 2)
    for r in range(8):
        for c in range(8):
            stone = board[r][c]
            center_x, center_y = BOARD_X + c * CELL_SIZE + CELL_SIZE // 2, BOARD_Y + r * CELL_SIZE + CELL_SIZE // 2
            if stone != 0:
                shadow_pos = (center_x + 3, center_y + 3)
                pygame.draw.circle(screen, STONE_SHADOW_COLOR, shadow_pos, STONE_RADIUS)
            if stone == 1: pygame.draw.circle(screen, BLACK, (center_x, center_y), STONE_RADIUS)
            elif stone == -1: pygame.draw.circle(screen, WHITE, (center_x, center_y), STONE_RADIUS)
    text_surface = font.render(message, True, INFO_COLOR)
    text_rect = text_surface.get_rect(center=(SCREEN_WIDTH // 2, BOARD_Y // 2 - 20))
    screen.blit(text_surface, text_rect)

def draw_menu():
    screen.fill(BG_COLOR)
    title_surface = font.render("Othello AI", True, WHITE)
    title_rect = title_surface.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
    screen.blit(title_surface, title_rect)
    h_vs_h_button = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 - 60, 300, 60)
    pygame.draw.rect(screen, BUTTON_BORDER_COLOR, h_vs_h_button, border_radius=10)
    pygame.draw.rect(screen, BUTTON_COLOR, h_vs_h_button.inflate(-6, -6), border_radius=7)
    h_vs_h_text = button_font.render("Human vs Human", True, BUTTON_TEXT_COLOR)
    h_vs_h_text_rect = h_vs_h_text.get_rect(center=h_vs_h_button.center)
    screen.blit(h_vs_h_text, h_vs_h_text_rect)
    h_vs_ai_button = pygame.Rect(SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 40, 300, 60)
    pygame.draw.rect(screen, BUTTON_BORDER_COLOR, h_vs_ai_button, border_radius=10)
    pygame.draw.rect(screen, BUTTON_COLOR, h_vs_ai_button.inflate(-6, -6), border_radius=7)
    h_vs_ai_text = button_font.render("Human vs AI", True, BUTTON_TEXT_COLOR)
    h_vs_ai_text_rect = h_vs_ai_text.get_rect(center=h_vs_ai_button.center)
    screen.blit(h_vs_ai_text, h_vs_ai_text_rect)
    return h_vs_h_button, h_vs_ai_button

# --- ✅ メイン関数 (バグ修正のため全面的に再構築) ---
def main():
    game = OthelloBoard()
    nnet, mcts = None, None
    game_state, board, current_player, game_over, message = "MENU", None, 1, False, ""
    
    clock = pygame.time.Clock()
    running = True

    while running:
        # 1. イベント処理 (常に全ての入力をここで受け取る)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # マウスクリックの処理
            if event.type == pygame.MOUSEBUTTONDOWN:
                if game_state == "MENU":
                    # メニュー画面でのクリック
                    if h_vs_h_button.collidepoint(event.pos):
                        game_state, board, current_player = "H_VS_H", game.getInitBoard(), 1
                        message = "Player 1's Turn (Black)"
                    elif h_vs_ai_button.collidepoint(event.pos):
                        if nnet is None:
                            nnet = NNetWrapper(game); nnet.load_checkpoint(folder="", filename=MODEL_FILE)
                            args = {'numMCTSSims': NUM_MCTS_SIMS, 'cpuct': 1.0}; mcts = MCTS(game, nnet, args)
                        game_state, board, current_player = "H_VS_AI", game.getInitBoard(), 1
                        message = "Your Turn (Black)"
                
                elif game_state in ["H_VS_H", "H_VS_AI"]:
                    # 人間の手番かどうかを判定
                    is_human_turn = (game_state == "H_VS_H") or (game_state == "H_VS_AI" and current_player == 1)
                    if is_human_turn and not game_over:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        # 盤面内でのクリックか判定
                        if BOARD_X <= mouse_x < BOARD_X + BOARD_SIZE and BOARD_Y <= mouse_y < BOARD_Y + BOARD_SIZE:
                            col, row = (mouse_x - BOARD_X) // CELL_SIZE, (mouse_y - BOARD_Y) // CELL_SIZE
                            action = row * 8 + col
                            valid_moves = game.getValidMoves(game.getCanonicalForm(board, current_player), 1)
                            if valid_moves[action]:
                                board, current_player = game.getNextState(board, current_player, action)
                                # メッセージを更新
                                if game_state == "H_VS_H": message = f"Player {current_player}'s Turn"
                                else: message = "AI is thinking..."
        
        # 2. ゲームロジックの更新 (AIの思考など)
        if game_state in ["H_VS_H", "H_VS_AI"] and not game_over:
            canonicalBoard = game.getCanonicalForm(board, current_player)
            valid_moves = game.getValidMoves(canonicalBoard, 1)

            # AIのターン処理
            if game_state == "H_VS_AI" and current_player == -1:
                if np.sum(valid_moves) > 0: # 置ける場所がある場合
                    draw_game(board, message); pygame.display.flip() # 「考え中」を表示
                    pi = mcts.getActionProb(canonicalBoard, temp=0)
                    action = np.argmax(pi)
                    board, current_player = game.getNextState(board, current_player, action)
                    message = "Your Turn (Black)"
                else: # パスの場合
                    board, current_player = game.getNextState(board, current_player, -1)
            
            # 人間同士のパスの処理
            elif np.sum(valid_moves) == 0:
                message = "Pass!"; draw_game(board, message); pygame.display.flip(); time.sleep(1)
                board, current_player = game.getNextState(board, current_player, -1)

            # ゲーム終了のチェック
            if game.getGameEnded(board, 1) != 0:
                game_over = True
                winner = game.getGameEnded(board, 1)
                if winner > 0: message = "Black Wins!"
                elif winner < 0: message = "White Wins!"
                else: message = "Draw!"
        
        # 3. 描画処理
        if game_state == "MENU":
            h_vs_h_button, h_vs_ai_button = draw_menu()
        else:
            draw_game(board, message)
        
        pygame.display.flip()
        
        # ゲーム終了後のリセット
        if game_over:
            time.sleep(3)
            game_state, game_over = "MENU", False
        
        clock.tick(30) # 1秒間に30フレームを上限とする

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
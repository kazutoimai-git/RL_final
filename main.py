import numpy as np
from tqdm import tqdm
import os
from collections import deque

from othello import OthelloBoard
from ai_model import OthelloNNet
from mcts import MCTS

import torch
import torch.optim as optim

args = {
    'numIters': 50, 'numEps': 100, 'tempThreshold': 15, 'updateThreshold': 0.55,
    'maxlenOfQueue': 200000, 'numMCTSSims': 100, 'arenaCompare': 40, 'cpuct': 1,
    'checkpoint': './temp/', 'load_model': False,
    'load_folder_file': ('./temp/','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
    'epochs': 10, 'batch_size': 64, 'cuda': torch.cuda.is_available(), 'lr': 0.001,
}

class NNetWrapper():
    def __init__(self, game):
        self.nnet = OthelloNNet(game)
        self.board_x, self.board_y = 8, 8
        self.action_size = game.getActionSize()
        if args['cuda']: self.nnet.cuda()

    def train(self, examples):
        optimizer = optim.Adam(self.nnet.parameters(), lr=args['lr'])
        for epoch in range(args['epochs']):
            print(f'EPOCH ::: {epoch+1}')

            self.nnet.train()

            pi_losses, v_losses = [], []
            batch_count = int(len(examples) / args['batch_size'])
            t = tqdm(range(batch_count), desc='Training NNet')
            for _ in t:
                sample_ids = np.random.randint(len(examples), size=args['batch_size'])
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))
                if args['cuda']:
                    boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()
                
                out_pi, out_v = self.nnet(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v
                pi_losses.append(float(l_pi)); v_losses.append(float(l_v))
                
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                t.set_postfix(pi_loss=np.mean(pi_losses), v_loss=np.mean(v_losses))

    def predict(self, board):
        board = torch.FloatTensor(board.astype(np.float64))
        if args['cuda']: board = board.contiguous().cuda()
        board = board.view(1, self.board_x, self.board_y)

        self.nnet.eval()

        with torch.no_grad():
            pi, v = self.nnet(board)
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0][0]

    def loss_pi(self, targets, outputs):
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets, outputs):
        return torch.sum((targets - outputs.view(-1))**2) / targets.size()[0]

    def save_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder): os.makedirs(folder)
        torch.save({'state_dict': self.nnet.state_dict()}, filepath)

    def load_checkpoint(self, folder='checkpoint', filename='checkpoint.pth.tar'):
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath): raise Exception(f"No model in path {filepath}")
        map_location = 'cuda:0' if args['cuda'] else 'cpu'
        checkpoint = torch.load(filepath, map_location=map_location)
        self.nnet.load_state_dict(checkpoint['state_dict'])

class Arena():
    def __init__(self, player1, player2, game):
        self.player1, self.player2, self.game = player1, player2, game

    def playGame(self):
        players = [self.player2, None, self.player1]
        curPlayer = 1
        board = self.game.getInitBoard()
        it = 0
        while self.game.getGameEnded(board, curPlayer) == 0:
            it += 1
            canonicalBoard = self.game.getCanonicalForm(board, curPlayer)
            # ✅ 修正済み: パスの状況を検知して処理
            if np.sum(self.game.getValidMoves(canonicalBoard, 1)) == 0:
                action = -1 # パス
            else:
                action = players[curPlayer + 1](canonicalBoard)
            
            board, curPlayer = self.game.getNextState(board, curPlayer, action)
        return self.game.getGameEnded(board, 1)

    def playGames(self, num):
        oneWon, twoWon, draws = 0, 0, 0
        for _ in tqdm(range(num // 2), desc="Arena.playGames (1)"):
            gameResult = self.playGame();
            if gameResult == 1: oneWon += 1
            elif gameResult == -1: twoWon += 1
            else: draws += 1
        self.player1, self.player2 = self.player2, self.player1
        for _ in tqdm(range(num - (num // 2)), desc="Arena.playGames (2)"):
            gameResult = self.playGame()
            if gameResult == -1: oneWon += 1
            elif gameResult == 1: twoWon += 1
            else: draws += 1
        return oneWon, twoWon, draws

class Coach():
    def __init__(self, game, nnet):
        self.game, self.nnet, self.pnet = game, nnet, NNetWrapper(game)
        self.trainExamplesHistory = []

    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        currentPlayer, episodeStep = 1, 0
        mcts = MCTS(self.game, self.nnet, args)
        
        while True:
            episodeStep += 1
            canonicalBoard = self.game.getCanonicalForm(board, currentPlayer)
            temp = int(episodeStep < args['tempThreshold'])
            
            # ✅ 修正済み: パスの状況を検知して処理
            if np.sum(self.game.getValidMoves(canonicalBoard, 1)) == 0:
                action = -1 # パス
            else:
                pi = mcts.getActionProb(canonicalBoard, temp=temp)
                sym = self.game.getSymmetries(canonicalBoard, pi)
                for b, p in sym:
                    trainExamples.append([b, currentPlayer, p, None])
                action = np.random.choice(len(pi), p=pi)

            board, currentPlayer = self.game.getNextState(board, currentPlayer, action)
            r = self.game.getGameEnded(board, 1)

            if r != 0:
                return [(x[0], x[2], r * ((-1)**(x[1] != 1))) for x in trainExamples]

    def learn(self):
        for i in range(1, args['numIters'] + 1):
            print(f'------ITER {i}------')
            iterationTrainExamples = deque([], maxlen=args['maxlenOfQueue'])
            for _ in tqdm(range(args['numEps']), desc="Self Play"):
                iterationTrainExamples.extend(self.executeEpisode())
            self.trainExamplesHistory.append(iterationTrainExamples)
            if len(self.trainExamplesHistory) > args['numItersForTrainExamplesHistory']: self.trainExamplesHistory.pop(0)
            
            trainExamples = [];
            for e in self.trainExamplesHistory: trainExamples.extend(e)
            
            self.nnet.save_checkpoint(folder=args['checkpoint'], filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=args.get('checkpoint'), filename='temp.pth.tar')
            self.nnet.train(trainExamples)
            
            print('PITTING AGAINST PREVIOUS VERSION')
            mcts1, mcts2 = MCTS(self.game, self.nnet, args), MCTS(self.game, self.pnet, args)
            arena = Arena(lambda x: np.argmax(mcts1.getActionProb(x, temp=0)),
                          lambda x: np.argmax(mcts2.getActionProb(x, temp=0)), self.game)
            nwins, pwins, draws = arena.playGames(args['arenaCompare'])

            print(f'NEW/PREV WINS : {nwins} / {pwins} ; DRAWS : {draws}')
            if pwins + nwins > 0 and float(nwins) / (pwins + nwins) < args['updateThreshold']:
                print('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=args['checkpoint'], filename='temp.pth.tar')
            else:
                print('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=args['checkpoint'], filename='best.pth.tar')

if __name__ == "__main__":
    game = OthelloBoard()
    nnet = NNetWrapper(game)
    if args['load_model']:
        nnet.load_checkpoint(args['load_folder_file'][0], args['load_folder_file'][1])
    c = Coach(game, nnet)
    c.learn()
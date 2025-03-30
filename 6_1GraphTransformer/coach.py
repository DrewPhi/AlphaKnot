import os
import numpy as np
from collections import deque
import random
import config
from arena import Arena
from mcts import MCTS
import time

class Coach:
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args

        self.trainExamplesHistory = deque([], maxlen=config.maxlenOfQueue)

        if config.load_model:
            self.loadTrainExamples()

    def executeEpisode(self):
        trainExamples = []
        board = self.game.getInitBoard()
        curPlayer = 1
        episodeStep = 0

        while True:
            episodeStep += 1
            canonicalBoard, current_player_canonical = self.game.getCanonicalForm(board, curPlayer)

            temp = int(episodeStep < config.tempThreshold)

            pi = MCTS(self.game, self.nnet, self.args).getActionProb(canonicalBoard, current_player_canonical, temp=temp)
            sym = [(canonicalBoard, pi, curPlayer)]
            trainExamples.extend(sym)

            action = np.random.choice(len(pi), p=pi)
            board, curPlayer = self.game.getNextState(board, curPlayer, action)

            r = self.game.getGameEnded(board, curPlayer)

            if r != 0:
                return [(x[0], x[1], r*((-1)**(x[2]!=curPlayer))) for x in trainExamples]

    def evaluate_against_random(self, nnet, num_games=100):
        """
        Evaluate the given nnet against a random player separately as Player 1 and Player 2.
        Returns the AI win rates going first and second.
        """
        from arena import Arena
        import numpy as np

        def nnet_player(board, player):
            canonicalBoard, current_player = self.game.getCanonicalForm(board, player)
            pi = MCTS(self.game, nnet, self.args).getActionProb(canonicalBoard, current_player, temp=0)
            return np.argmax(pi)



        def random_player(board, player):
            valids = self.game.getValidMoves(board, player).cpu().numpy()
            valid_actions = np.where(valids == 1)[0]
            return np.random.choice(valid_actions)

        # AI as Player 1 (going first)
        arena_p1 = Arena(nnet_player, random_player, self.game)
        ai_p1_wins, random_p2_wins, draws_p1 = arena_p1.playGames(num_games)

        # AI as Player 2 (going second)
        arena_p2 = Arena(random_player, nnet_player, self.game)
        random_p1_wins, ai_p2_wins, draws_p2 = arena_p2.playGames(num_games)

        # Calculate percentages separately
        ai_p1_winrate = (ai_p1_wins / num_games) * 100
        ai_p2_winrate = (ai_p2_wins / num_games) * 100
        draw_rate_p1 = (draws_p1 / num_games) * 100
        draw_rate_p2 = (draws_p2 / num_games) * 100

        print("\n🎲 Evaluation vs Random Player:")
        print(f"➡️  AI going FIRST: AI wins: {ai_p1_winrate:.2f}%, Draws: {draw_rate_p1:.2f}%")
        print(f"⬅️  AI going SECOND: AI wins: {ai_p2_winrate:.2f}%, Draws: {draw_rate_p2:.2f}%\n")

        # Explicitly return these values!
        return ai_p1_winrate, ai_p2_winrate


    def learn(self):
        total_start = time.time()
        for i in range(1, config.numIters+1):
            print(f'\n{"="*30}\n🚀 STARTING ITERATION {i}/{config.numIters}\n{"="*30}')
            
            iteration_start = time.time()

            iterationTrainExamples = []
            for ep in range(1, config.numEps + 1):
                ep_start = time.time()
                iterationTrainExamples += self.executeEpisode()
                ep_time = time.time() - ep_start
                print(f'[Self-play] Episode {ep}/{config.numEps} completed in {ep_time:.2f}s')

            self.trainExamplesHistory.extend(iterationTrainExamples)
            self.saveTrainExamples(i-1)

            random.shuffle(iterationTrainExamples)
            print("[Training] Started neural network training.")
            train_start = time.time()
            self.nnet.train(iterationTrainExamples)
            train_time = time.time() - train_start
            print(f"[Training] Completed training in {train_time:.2f}s")

            if config.saveIterCheckpoint:
                folder = config.checkpoint
                filename = f'checkpoint_{i}.pth.tar'
                self.nnet.save_checkpoint(os.path.join(folder, filename))
                print(f"[Checkpoint] Saved: {filename}")
            if i == 1:
                print('🥇 Initial iteration complete; setting current model as champion.')
                self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))

            if config.arenaCompare > 0 and i > 1:
                print("[Arena] Evaluating against previous model...")
                prev_nnet = self.nnet.__class__(self.game)
                prev_nnet.load_checkpoint(os.path.join(config.checkpoint, f'checkpoint_{i-1}.pth.tar'))

                # Alternate roles fairly
                def player_fn(nnet):
                    return lambda board, player: np.argmax(MCTS(self.game, nnet, self.args)
                                                        .getActionProb(*self.game.getCanonicalForm(board, player), temp=0))


                arena1 = Arena(player_fn(self.nnet), player_fn(prev_nnet), self.game)
                nwins1, pwins1, draws1 = arena1.playGames(config.arenaCompare // 2)

                arena2 = Arena(player_fn(prev_nnet), player_fn(self.nnet), self.game)
                pwins2, nwins2, draws2 = arena2.playGames(config.arenaCompare // 2)

                nwins = nwins1 + nwins2
                pwins = pwins1 + pwins2
                draws = draws1 + draws2

                winRate = float(nwins) / (pwins + nwins) if (pwins + nwins) else 0

                print(f'[Arena Results] New model wins: {nwins}, Previous model wins: {pwins}, Draws: {draws}')
                print(f'[Arena Results] New model win rate: {winRate:.2%}')

                # Evaluate both models against Random
                print("\n📌 Evaluating CURRENT CANDIDATE vs Random Player:")
                ai_current_p1, ai_current_p2 = self.evaluate_against_random(self.nnet, num_games=100)

                print("\n📌 Evaluating PREVIOUS CHAMPION vs Random Player:")
                ai_prev_p1, ai_prev_p2 = self.evaluate_against_random(prev_nnet, num_games=100)

                current_avg = (ai_current_p1 + ai_current_p2) / 2
                prev_avg = (ai_prev_p1 + ai_prev_p2) / 2

                # Tie-breaking based on average random win rate
                if winRate > config.updateThreshold:
                    print('🎉 Accepting new model as champion (Arena winrate).')
                    self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                elif winRate == 0.5:  # Exactly tied, use random win rate as tiebreaker
                    if current_avg > prev_avg:
                        print('⚖️ Arena tied; choosing NEW model based on higher average random win rate.')
                        self.nnet.save_checkpoint(os.path.join(config.checkpoint, 'best.pth.tar'))
                    else:
                        print('⚖️ Arena tied; choosing PREVIOUS model based on higher average random win rate.')
                        self.nnet.load_checkpoint(os.path.join(config.checkpoint, f'checkpoint_{i-1}.pth.tar'))
                else:
                    print('❌ Rejecting new model. Reverting to previous best.')
                    self.nnet.load_checkpoint(os.path.join(config.checkpoint, f'checkpoint_{i-1}.pth.tar'))


            iteration_time = (time.time() - iteration_start)/60
            total_elapsed = (time.time() - total_start)/60
            est_remaining = (total_elapsed / i) * (config.numIters - i)
            print(f"[Iteration] Iteration time: {iteration_time:.2f} minutes | "
                f"Total elapsed: {total_elapsed:.2f} minutes | "
                f"Estimated remaining: {est_remaining:.2f} minutes")


    def saveTrainExamples(self, iteration):
        folder = config.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, f'train_examples_{iteration}.pkl')
        with open(filename, "wb") as f:
            import pickle
            pickle.dump(self.trainExamplesHistory, f)

    def loadTrainExamples(self):
        filename = os.path.join(config.load_folder_file[0], config.load_folder_file[1]+".examples")
        if not os.path.isfile(filename):
            print(f"File {filename} with trainExamples not found!")
        else:
            print("File with trainExamples found. Loading...")
            with open(filename, "rb") as f:
                import pickle
                self.trainExamplesHistory = pickle.load(f)
            print('Loading done!')
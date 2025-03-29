# arena.py
import numpy as np

class Arena:
    """
    An Arena class where two agents (functions returning actions) can be pitted against each other.
    Modeled after Surag's alpha-zero-general approach.
    """

    def __init__(self, player1, player2, game, display=None):
        """
        player1, player2: functions that take 'board' and return an action (integer).
        game: the game object (KnotGraphGame).
        display: optional function to render the board (not mandatory).
        """
        self.player1 = player1
        self.player2 = player2
        self.game = game
        self.display = display

    def playGame(self, verbose=False):
        """
        Plays one game between player1 and player2. 
        Returns:
            +1 if player1 wins,
            -1 if player2 wins,
             1e-4 or something if it's a draw (not likely in this game).
        """
        currentPlayer = 1
        board = self.game.getInitBoard()

        while True:
            if verbose and self.display:
                self.display(board)

            if currentPlayer == 1:
                action = self.player1(board)
            else:
                action = self.player2(board)

            board, currentPlayer = self.game.getNextState(board, currentPlayer, action)

            result = self.game.getGameEnded(board, currentPlayer)
            if result != 0:
                return result * currentPlayer  # Flip sign so +1 means player1 wins, -1 means player2 wins

    def playGames(self, num, verbose=False):
        """
        Plays 'num' games where player1 always starts first.
        Returns:
            (score, total) where 'score' is how many games player1 won out of 'num'.
        """
        oneWon = 0
        twoWon = 0
        draws = 0
        for _ in range(num):
            gameResult = self.playGame(verbose=verbose)
            if gameResult == 1:
                oneWon += 1
            elif gameResult == -1:
                twoWon += 1
            else:
                draws += 1

        return oneWon, twoWon, draws

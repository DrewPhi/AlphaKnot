"""Explicit spawn-based arena smoke test."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arena import Arena
from knot_graph_game import KnotGraphGame
from knot_graph_nnet import NNetWrapper


class SmokeArgs:
    numMCTSSims = 1
    cpuct = 1.0


def main():
    game = KnotGraphGame()
    game.getInitBoard()
    network = NNetWrapper(game, device="cpu")

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = str(Path(directory) / "smoke.pth.tar")
        network.save_checkpoint(checkpoint)
        results = Arena(
            nnet1_path=checkpoint,
            nnet2_path=checkpoint,
            game_class=KnotGraphGame,
            args=SmokeArgs(),
        ).playGames_parallel(num_games=2, num_workers=2)

    if sum(results) != 2:
        raise RuntimeError(f"Unexpected arena results: {results}")
    print(f"arena smoke test passed: {results}")


if __name__ == "__main__":
    main()

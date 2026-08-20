"""Explicit spawn-based self-play smoke test.

Run from the repository root with:
``python tests/smoke_multiprocessing.py``.
"""

import sys
import tempfile
from multiprocessing import get_context
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coach import _init_worker, _run_episode
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
        context = get_context("spawn")
        with context.Pool(
            processes=2,
            initializer=_init_worker,
            initargs=(checkpoint, SmokeArgs()),
        ) as pool:
            results = pool.map(_run_episode, (1, 2))

    if [len(episode) for episode in results] != [7, 7]:
        raise RuntimeError("Unexpected self-play episode length")
    print("multiprocessing smoke test passed")


if __name__ == "__main__":
    main()

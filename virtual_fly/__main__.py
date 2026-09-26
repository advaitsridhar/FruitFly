"""``python -m virtual_fly`` runs the brain command line; ``python -m virtual_fly.play`` (or ``python fly_game.py``) the game."""
import sys

from .cli import main

sys.argv[0] = "python -m virtual_fly"     # argparse names the program after it (else "__main__.py", before Python 3.14)
main()

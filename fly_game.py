#!/usr/bin/env python3
"""
fly_game.py - play with a virtual fly whose every move comes out of a simulated fly brain.

    pip install numpy
    python fly_game.py              # then play in the browser tab that opens
    python fly_game.py --help       # options (port, model profile, ...)

The brain is the whole male fruit fly central nervous system (176,422 neurons), simulated live by
the ``virtual_fly`` package next to this file. This entry point just starts the game.
"""
from virtual_fly.play import main

if __name__ == "__main__":
    main()

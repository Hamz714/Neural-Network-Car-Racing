"""Neural-network car racing.

A 2D racing game whose opponents are driven by a feed-forward neural network
written from scratch in pure Python, trained by a genetic algorithm.

Layout:
    neural_network  the network itself - pure Python, no third-party imports
    entities        Car / PlayerCar / NPC / Track / Sensor / Checkpoint
    ui              buttons and the frame-counted Event timer
    game            per-frame helpers shared by the game loop
    screens         the interactive game's menus and race loop
    sim             headless simulation used for training (numpy permitted)
    ga              the genetic algorithm
"""

__version__ = "1.0.0"

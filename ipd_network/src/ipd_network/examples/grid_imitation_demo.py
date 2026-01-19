"""Grid imitation demo for the Iterated Prisoner's Dilemma.

Each node starts with a random action (C or D). At each step, nodes copy the
neighbor with the highest payoff (ties broken randomly). The grid is periodic
so every node has four neighbors.
"""

import logging

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap

from ipd_network.simulation import GridImitationModel
from ipd_network.utils import get_logger

logger = get_logger(__name__, level=logging.INFO)


def run_grid_imitation_demo(size=20, steps=50, seed=42, interval=300):
    """Run a grid imitation demo with animated visualization."""
    # The model handles random initialization and the imitation rule.
    model = GridImitationModel(size=size, rounds=steps, seed=seed)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = ListedColormap(["#e6f2e4", "#d1495b"])
    image = ax.imshow(model.action_grid(), cmap=cmap, vmin=0, vmax=1)
    ax.set_title("IPD Grid Imitation (C = light, D = red)")
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        model.step()
        image.set_data(model.action_grid())
        ax.set_xlabel(f"Step {frame + 1}/{steps}")
        return [image]

    logger.info("Starting grid imitation demo for %s steps", steps)
    animation = FuncAnimation(fig, update, frames=steps, interval=interval, blit=True)
    plt.show()
    return animation


if __name__ == "__main__":
    run_grid_imitation_demo()

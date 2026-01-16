import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# Prisoner's Dilemma parameters
# -----------------------------
R = 1.0  # reward for mutual cooperation
P = 0.0  # punishment for mutual defection
S = 0.0  # sucker's payoff (C vs D)
T = 1.6  # temptation payoff (D vs C)  (Mesa example often uses ~1.6)


def payoff(a, b):
    """Return payoff for player with action a against opponent b.
    a, b are 0 (C) or 1 (D).
    """
    if a == 0 and b == 0:
        return R
    if a == 0 and b == 1:
        return S
    if a == 1 and b == 0:
        return T
    return P


def neighbors(i, j, n):
    """Moore neighborhood (8 neighbors) on a torus."""
    out = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            out.append(((i + di) % n, (j + dj) % n))
    return out


def step(grid):
    """One synchronous update step:
    - compute each cell's total score vs its neighbors
    - each cell adopts the strategy of the best-scoring neighbor (incl itself)
    """
    n = grid.shape[0]
    scores = np.zeros((n, n), dtype=float)

    # score accumulation
    for i in range(n):
        for j in range(n):
            a = grid[i, j]
            s = 0.0
            for ni, nj in neighbors(i, j, n):
                b = grid[ni, nj]
                s += payoff(a, b)
            scores[i, j] = s

    # update: imitate best neighbor (including self)
    new_grid = grid.copy()
    for i in range(n):
        for j in range(n):
            best_i, best_j = i, j
            best_score = scores[i, j]

            for ni, nj in neighbors(i, j, n):
                if scores[ni, nj] > best_score:
                    best_score = scores[ni, nj]
                    best_i, best_j = ni, nj

            new_grid[i, j] = grid[best_i, best_j]

    return new_grid


def run_animation(n=60, p_defect=0.5, steps=200, interval_ms=60):
    # 0 = cooperate, 1 = defect
    grid = (np.random.rand(n, n) < p_defect).astype(int)

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title("Prisoner's Dilemma on a Grid (0=C, 1=D)")
    ax.set_xticks([])
    ax.set_yticks([])

    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="white",
        bbox=dict(facecolor="black", alpha=0.5, pad=4),
    )

    def update(frame):
        nonlocal grid
        grid = step(grid)
        im.set_data(grid)
        coop_frac = np.mean(grid == 0)
        text.set_text(f"t={frame:03d}   cooperation={coop_frac:.3f}")
        return im, text

    ani = FuncAnimation(fig, update, frames=steps, interval=interval_ms, blit=False)
    plt.show()


if __name__ == "__main__":
    run_animation(
        n=60,  # grid size
        p_defect=0.5,  # initial fraction of defectors
        steps=300,  # number of steps
        interval_ms=60,
    )

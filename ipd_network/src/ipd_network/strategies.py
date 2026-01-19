"""Strategy helpers using the Axelrod library."""

import axelrod as axl

STRATEGIES = {
    "cooperator": axl.Cooperator,
    "defector": axl.Defector,
    "tit_for_tat": axl.TitForTat,
    "grim": axl.Grudger,
    "random": axl.Random,
}


def available_strategies():
    """Return a curated subset of available Axelrod strategies."""
    return dict(STRATEGIES)


def create_strategy(name):
    """Instantiate an Axelrod strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}")
    return STRATEGIES[name]()

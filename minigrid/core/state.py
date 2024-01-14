from dataclasses import dataclass, field
from minigrid.core.constants import (
    COLOR_NAMES,
    IDX_TO_COLOR
)

@dataclass(frozen=True, eq=True)
class Key:
    color: str = ""
    col: int = 0
    row: int = 0

@dataclass(frozen=True, eq=True)
class Ball:
    color: str = ""
    col: int = 0
    row: int = 0

@dataclass(frozen=True, eq=True)
class Box:
    color: str = ""
    col: int = 0
    row: int = 0

@dataclass(frozen=True, eq=True)
class Adversary:
    color: str = ""
    col: int = 0
    row: int = 0
    view: int = 0

@dataclass(frozen=True, eq=True)
class State:
    colAgent: int
    rowAgent: int
    viewAgent: int
    adversaries: tuple = field(default_factory=tuple)
    balls: tuple = field(default_factory=tuple)

def to_state(ints, booleans):
    agentState = (ints["colAgent"], ints["rowAgent"], ints["viewAgent"])
    ints = {key:int(value) for key, value in ints.items()}
    booleans = {value: False if key == "!" else True for key, value in booleans.items()}
    adversaries = tuple()
    boxes = tuple()
    balls = tuple()
    keys = tuple()
    for color in COLOR_NAMES:
        color = color.capitalize()
        if "col" + color in ints:
            adversaries += (Adversary(color, ints["col"+color], ints["row"+color], ints["view"+color]),)
        if "col" + color + "Ball" in ints:
            identifier = color + "Ball"
            balls += (Ball(color, ints["col"+identifier], ints["row"+identifier]),)
        if "col" + color + "Box" in ints:
            pass
        if "col" + color + "Key" in ints:
            pass


    return State(*agentState, adversaries, balls)

from dataclasses import dataclass, field

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
    adversaries: tuple

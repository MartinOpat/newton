
from __future__ import annotations
from typing import Optional, Tuple, List

import math
import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State
from ..solver import SolverBase
from .kernels import (
    eval_body_joint_forces,
    eval_muscle_forces,
)


def _sigmoid(d: float, kappa: float) -> float:
    # Eq. (5) in paper
    # NOTE: guard against overflow for large kappa*|d|
    x = max(-50.0, min(50.0, d * kappa))
    return 1.0 / (1.0 + math.exp(-x))



class SolverMoreau(SolverBase):
    """
    Moreau time stepping
    """

    def __init__(
        self,
        model: Model,
        gs_iterations: int = 10,
        kappa: float = 300.0,
        beta: float = 0.2,
        restitution: float = 0.0,
        friction_smoothing: float = 1.0,
        angular_damping: float = 0.0,
        activation_distance: float = float("inf"),
        use_world_moment_arm: bool = True,
    ):
        super().__init__(model=model)
        self.gs_iterations = gs_iterations
        self.kappa = kappa
        self.beta = beta
        self.restitution = restitution
        self.friction_smoothing = friction_smoothing
        self.angular_damping = angular_damping
        self.activation_distance = activation_distance
        self.use_world_moment_arm = use_world_moment_arm


    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Optional[Control],
        contacts: Optional[Contacts],
        dt: float,
    ):
        pass

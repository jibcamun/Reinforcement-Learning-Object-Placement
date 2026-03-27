"""
App Environment Implementation.
"""

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import AppAction, AppObservation, AppState
except ImportError:
    from models import AppAction, AppObservation, AppState

from utils import *


OBJECTS = {
    "book": {"dims": [4, 4, 2], "stack": True},
    "penstand": {"dims": [2, 2, 4], "stack": True},
    "bottle": {"dims": [2, 2, 6], "stack": False},
    "pen": {"dims": [1, 1, 4], "stack": False},
    "pencil": {"dims": [1, 1, 6], "stack": False},
    "eraser": {"dims": [2, 1, 1], "stack": False},
    "powerbank": {"dims": [4, 2, 1], "stack": False},
    "mobile": {"dims": [4, 2, 1], "stack": False},
    "laptop": {"dims": [6, 4, 1], "stack": True},
    "monitor": {"dims": [6, 4, 2], "stack": False},
    "keyboard": {"dims": [6, 2, 1], "stack": False},
    "mouse": {"dims": [4, 2, 1], "stack": False},
    "headphones": {"dims": [4, 4, 2], "stack": False},
    "charger": {"dims": [2, 2, 1], "stack": False},
    "notebook": {"dims": [4, 4, 1], "stack": True},
    "folder": {"dims": [4, 4, 1], "stack": True},
    "backpack": {"dims": [6, 4, 2], "stack": False},
    "pouch": {"dims": [4, 4, 2], "stack": False},
}


class AppEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> AppObservation:
        self._state = AppState(
            episode_id=str(uuid4()),
            step_count=0,
            currentGrid=initGrid(),
            weightedGrid=initWeightedGrid(),
            objectsLeft=list(OBJECTS.keys()),
            objectsFound=[],
            reward=0.0,
            isDone=False,
            ObjectsPresent={},
        )

        return AppObservation(
            currentGrid=self._state.currentGrid,
            positions=self._state.ObjectsPresent,
            objectsLeft=self._state.objectsLeft,
            objectsFound=self._state.objectsFound,
        )

    def step(self, action: AppAction) -> AppObservation:
        self._state.step_count += 1

        reward = 0.0

        if action.placement:
            reward += place(action.placement, self._state)

        if action.find_objects:
            reward += findobject(action.findObjects, self._state)

        if len(self._state.objectsLeft) == 0:
            self._state.isDone = True
            reward += 100

        self._state.reward += reward

        return AppObservation(
            currentGrid=self._state.currentGrid,
            positions=self._state.ObjectsPresent,
            objectsLeft=self._state.objectsLeft,
            objectsFound=self._state.objectsFound,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

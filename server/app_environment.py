from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import AppAction, AppObservation, AppState
except ImportError:
    from models import AppAction, AppObservation, AppState

try:
    from ..utils import *
except ImportError:
    from utils import *


class AppEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = self._new_state()
        self._reset_count = 0

    def _new_state(self) -> AppState:
        grid, placed = initGrid()

        return AppState(
            episode_id=str(uuid4()),
            step_count=0,
            currentGrid=grid,
            weightedGrid=initWeightedGrid(),
            objectsLeft=list(OBJECTS.keys()),
            objectsFound=[],
            reward=0.0,
            isDone=False,
            ObjectsPresent=placed,
        )

    def reset(self) -> AppObservation:
        self._state = self._new_state()

        return AppObservation(
            currentGrid=self._state.currentGrid,
            positions=self._state.ObjectsPresent,
            objectsLeft=self._state.objectsLeft,
            objectsFound=self._state.objectsFound,
            reward=self._state.reward,
            isDone=self._state.isDone,
        )

    def step(self, action: AppAction) -> AppObservation:
        if not isinstance(self._state, AppState):
            self._state = self._new_state()

        self._state.step_count += 1

        reward = 0.0

        if action.placement:
            reward += place(action.placement, self._state)

        if action.findObjects:
            reward += findobject(action.findObjects, self._state)

        if len(self._state.objectsLeft) == 0:
            self._state.isDone = True
            reward += 100

        self._state.reward += reward / (10**self._state.step_count)

        return AppObservation(
            currentGrid=self._state.currentGrid,
            positions=self._state.ObjectsPresent,
            objectsLeft=self._state.objectsLeft,
            objectsFound=self._state.objectsFound,
            reward=self._state.reward,
            isDone=self._state.isDone,
        )

    @property
    def state(self) -> dict:
        return self._state.model_dump()

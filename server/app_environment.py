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

    def _coerce_state(self) -> AppState:

        if isinstance(self._state, AppState):
            return self._state

        if isinstance(self._state, dict):
            self._state = AppState(**self._state)
            return self._state

        self._state = self._new_state()
        return self._state

    def _new_state(self) -> AppState:
        grid, placed = initGrid()
        grid_shape = (len(grid), len(grid[0]), len(grid[0][0]))

        return AppState(
            episode_id=str(uuid4()),
            step_count=0,
            currentGrid=grid,
            weightedGrid=initWeightedGrid(grid_shape),
            objectsLeft=list(placed.keys()),
            objectsFound=[],
            reward=0.0,
            isDone=False,
            ObjectsPresent=placed,
            ObjectsPlaced={},
            rewardFeedback=[],
            rewardList=[],
            numberPlaced=0,
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
            rewardFeedback=self._state.rewardFeedback,
            rewardList=self._state.rewardList,
            numberPlaced=self._state.numberPlaced,
            ObjectsPlaced=self._state.ObjectsPlaced,
        )

    def step(self, action: AppAction) -> AppObservation:
        state = self._coerce_state()

        if isinstance(action, dict):
            action = AppAction(**action)

        state.step_count += 1
        reward = 0.0

        if action is None:
            reward -= 10.0
            appendRewardFeedback(
                state,
                "No action is of invalid schema or format. Penalty applied.",
                reward,
            )
            return AppObservation(
                currentGrid=state.currentGrid,
                positions=state.ObjectsPresent,
                objectsLeft=state.objectsLeft,
                objectsFound=state.objectsFound,
                reward=state.reward,
                isDone=state.isDone,
                rewardFeedback=state.rewardFeedback,
                rewardList=state.rewardList,
                numberPlaced=state.numberPlaced,
                ObjectsPlaced=state.ObjectsPlaced,
            )

        if action.isSegmentation and action is not None:
            reward += 10.0
            appendRewardFeedback(state, "Segmentation successful.", reward)

        if action.placement and action is not None:
            placement_reward, placement_failed = place(
                action.isSegmentation, action.placement, state
            )
            reward += placement_reward
            if placement_failed:
                appendRewardFeedback(state, "Failed to place object.", reward)
            else:
                appendRewardFeedback(state, "Object placed successfully.", reward)

        if action.adjust and action is not None:
            reward += adjustment(action.isSegmentation, action.adjust, state)
            appendRewardFeedback(state, "Object adjusted successfully.", reward)

        if action.findObjects and action is not None:
            reward += findobject(action.isSegmentation, action.findObjects, state)
            appendRewardFeedback(state, "Object found successfully.", reward)

        if len(state.objectsLeft) == 0 and state.ObjectsPresent == state.numberPlaced:
            state.isDone = True
            reward += 10.0
            appendRewardFeedback(state, "All objects found. Episode completed!", reward)

        state.reward += reward / (10**state.step_count)

        return AppObservation(
            currentGrid=state.currentGrid,
            positions=state.ObjectsPresent,
            objectsLeft=state.objectsLeft,
            objectsFound=state.objectsFound,
            reward=state.reward,
            isDone=state.isDone,
            rewardFeedback=state.rewardFeedback,
            rewardList=state.rewardList,
            numberPlaced=state.numberPlaced,
            ObjectsPlaced=state.ObjectsPlaced,
        )

    @property
    def state(self) -> dict:
        state = self._coerce_state()
        return state.model_dump()

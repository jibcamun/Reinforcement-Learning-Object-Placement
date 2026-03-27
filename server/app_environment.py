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
    """
    A simple echo environment that echoes back messages.

    This environment is designed for testing the HTTP server infrastructure.
    It maintains minimal state and simply echoes back whatever message it receives.

    Example:
        >>> env = AppEnvironment()
        >>> obs = env.reset()
        >>> print(obs.echoed_message)  # "App environment ready!"
        >>>
        >>> obs = env.step(AppAction(message="Hello"))
        >>> print(obs.echoed_message)  # "Hello"
        >>> print(obs.message_length)  # 5
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the app environment."""
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(self) -> AppObservation:
        """
        Reset the environment.

        Returns: AppObservation
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1

        return AppObservation(
            echoed_message="App environment ready!",
            message_length=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: AppAction) -> AppObservation:
        """
        Execute a step in the environment by echoing the message.

        Args:
            action: AppAction

        Returns:
            AppObservation
        """

        self._state.step_count += 1

        if action.isSegmentation and self._state.step_count == 1:
            reward = 10

        if action.placement:
            place(action.placement, self._state)
        elif action.findobjects:
            findobject(action, self._state)

        return AppObservation(
            currentGrid=[],
            postions={},
            objectsLeft=[],
            objectsFound=[],
            reward=reward,
            isDone=False,
        )

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state

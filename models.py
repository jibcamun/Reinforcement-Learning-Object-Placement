from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import List, Dict, Tuple


class AppAction(Action):
    """Action for the App environment"""

    placement: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict, description="Placement of the object in a 3D grid"
    )
    isSegmentation: bool = Field(
        default=True, description="Whether the model is segmenting the objects"
    )
    findObjects: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict, description="Dictionary of objects"
    )


class AppObservation(Observation):
    """Observation from the App environment"""

    currentGrid: List[List[List[int]]] = Field(
        default_factory=list,
        description="Current placement of the objects in a 3D grid",
    )
    positions: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict,
        description="Dictionary of objects with their positions in the environment",
    )
    objectsLeft: List[str] = Field(
        default_factory=list,
        description="List of unorganised objects left in the environment",
    )
    objectsFound: List[str] = Field(
        default_factory=list,
        description="List of objects found in the environment",
    )
    reward: float = Field(
        default=0.0, description="Reward received after taking the action"
    )
    isDone: bool = Field(default=False, description="Whether the episode has ended")


class AppState(State):
    """State for the App environment"""

    currentGrid: List[List[List[int]]] = Field(
        default_factory=list,
        description="Initial state of the environment with unorganised objects",
    )

    weightedGrid: List[List[List[float]]] = Field(
        default_factory=list,
        description="Weighted grid used when scoring placements",
    )

    objectsLeft: List[str] = Field(
        default_factory=list,
        description="List of unorganised objects left in the environment",
    )
    objectsFound: List[str] = Field(
        default_factory=list,
        description="List of objects found in the environment",
    )
    reward: float = Field(
        default=0.0, description="Reward received after taking the action"
    )
    isDone: bool = Field(default=False, description="Whether the episode has ended")

    ObjectsPresent: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict,
        description="Placed objects and their current positions in the environment",
    )

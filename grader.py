try:
    from models import AppObservation, AppState
except ImportError:
    from app.models import AppObservation, AppState

#
# currentGrid: List[List[List[int]]] = Field(
#    default_factory=list,
#    description="Current placement of the o
# )
#
#
# positions: Dict[str, Tuple[int, int, int, b
#    default_factory=dict,
#    description="Dictionary of objects with
# )
#
#
# objectsLeft: List[str] = Field(
#    default_factory=list,
#    description="List of unorganised object
# )
#
#
# objectsFound: List[str] = Field(
#    default_factory=list,
#    description="List of objects found in t
# )
#
#
# reward: float = Field(
#    default=0.0, description="Reward receiv
# )
#
#
# isDone: bool = Field(default=False, descrip
#
#
# rewardFeedback: list[str] = Field(
#    default_factory=list,
#    description="List of feedback strings d
# )
#
#
# rewardList: list[float] = Field(
#    default_factory=list,
#    description="List of reward values rece
# )
#
#
# numberPlaced: int = Field(
#    default=0,
#    description="Number of objects successf
# )
#
#
# ObjectsPlaced: Dict[str, Tuple[int, int, in
#    default_factory=dict,
#    description="Objects that have been suc
# )


def segmentation(history, state):
    pass

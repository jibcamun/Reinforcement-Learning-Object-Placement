"""App Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AppAction, AppObservation


class AppEnv(EnvClient[AppAction, AppObservation, State]):

    def _step_payload(self, action: AppAction) -> Dict:

        return {
            "placement": action.placement,
            "isSegmentation": action.isSegmentation,
            "findObjects": action.findObjects,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AppObservation]:

        obs_data = payload.get("observation", {})
        observation = AppObservation(
            currentGrid=obs_data.get("currentGrid", []),
            postions=obs_data.get("postions", {}),
            objectsLeft=obs_data.get("objectsLeft", []),
            objectsFound=obs_data.get("objectsFound", []),
            reward=obs_data.get("reward", 0.0),
            isDone=obs_data.get("isDone", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("isDone", False),
        )

    def _parse_state(self, payload: Dict) -> State:

        return State(
            currentGrid=payload.get("currentGrid", []),
            weightedGrid=payload.get("weightedGrid", []),
            reward=payload.get("reward", 0.0),
            isDone=payload.get("isDone", False),
            objectsLeft=payload.get("objectsLeft", []),
            objectsFound=payload.get("objectsFound", []),
            ObjectsPresent=payload.get("ObjectsPresent", {}),
        )

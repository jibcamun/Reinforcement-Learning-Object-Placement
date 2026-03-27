"""App Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from .models import AppAction, AppObservation, AppState


class AppEnv(EnvClient[AppAction, AppObservation, AppState]):

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
            positions=obs_data.get("positions", {}),
            objectsLeft=obs_data.get("objectsLeft", []),
            objectsFound=obs_data.get("objectsFound", []),
            reward=obs_data.get("reward", 0.0),
            isDone=obs_data.get("isDone", False),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", obs_data.get("isDone", False)),
        )

    def _parse_state(self, payload: Dict) -> AppState:

        return AppState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            currentGrid=payload.get("currentGrid", []),
            weightedGrid=payload.get("weightedGrid", []),
            reward=payload.get("reward", 0.0),
            isDone=payload.get("isDone", False),
            objectsLeft=payload.get("objectsLeft", []),
            objectsFound=payload.get("objectsFound", []),
            ObjectsPresent=payload.get("ObjectsPresent", {}),
        )

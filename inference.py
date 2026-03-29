import os
from dotenv import load_dotenv
from openai import OpenAI
import json

try:
    from models import AppAction, AppObservation
except ImportError:
    from app.models import AppAction, AppObservation

try:
    from server.app_environment import AppEnvironment
except ImportError:
    from app.server.app_environment import AppEnvironment


load_dotenv()

API_URL = os.getenv("API_BASE_URL")
MODEL = os.getenv("MODEL_NAME")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")

MAX_STEPS = 8
TEMPERATURE = 0.2
MAX_TOKENS = 500
FALLBACK_ACTION = "findObjects()"

DEBUG = True

SYSTEM_PROMPT = """
        You are an intelligent agent controlling a 3D object placement environment. Your task is to:

        1. **Segment objects** in the environment if `isSegmentation=True`.
        2. **Identify objects** and their properties (name, stackable) accurately.
        3. **Place objects** in the 3D grid respecting stacking rules and dimensions.
        4. **Use rewards and feedback** from previous steps to improve future actions.

        You must strictly return actions that conform to this Pydantic schema:

        AppAction:
        {
            placement: Dict[str, Tuple[int, int, int, bool]]  # object_name -> (x, y, z, stackable)
            isSegmentation: bool
            findObjects: Dict[str, Tuple[int, int, int, bool]]  # object_name -> (x, y, z, stackable)
        }

        Rules:
        - Only report objects that are found or placed; empty dicts are valid if none.
        - Do not modify objects that are already placed unless instructed.
        - Coordinates must be within the grid bounds.
        - Respect stackable property: non-stackable objects cannot be placed on top of another object.
        - Use previous step’s reward and rewardFeedback to adjust your strategy.

        Output:
        - Always return a valid JSON object conforming to the schema.
        - Do not include any extra text, explanations, or commentary.
        - If no action is possible, return empty dicts for `placement` and `findObjects`.

        Your goal:
        - Maximize cumulative reward.
        - Identify all objects correctly.
        - Place objects efficiently while respecting stacking rules.
        - Learn from reward feedback to improve placement in future steps.

        Always return a valid JSON that conforms exactly to the AppAction Pydantic model:
        - placement: Dict[str, Tuple[int,int,int,bool]] or null
        - isSegmentation: bool
        - findObjects: Dict[str, Tuple[int,int,int,bool]] or null
        
        Actions:
        - To place an object: {"isSegmentation": false, "placement": {"object_name": (x, y, z, stackable)}, "findObjects": {}}
        - To segment objects: {"isSegmentation": true, "placement": {}, "findObjects": {"object_name": (x, y, z, stackable)}}
        
        Do not include explanations, text, or extra fields.
        If no objects are found or placed, return empty dicts for placement and findObjects.
        The output must be parseable and valid for AppAction(**json_output).
    """.strip()

MESSAGES = [{"role": "system", "content": SYSTEM_PROMPT}]
HISTORY = []


def parse_output(output_str: str) -> AppAction:
    try:
        data = json.loads(output_str)
        action = AppAction(**data)
        return action
    except:
        print("Invalid Output")


def main() -> None:
    env = AppEnvironment()
    observation: AppObservation = env.reset()

    client = OpenAI(
        base_url=API_URL,
        api_key=API_KEY,
    )
    for i in range(1, MAX_STEPS + 1):
        MESSAGES.append(
            {
                "role": "user",
                "content": f"""Observation: {observation.model_dump_json()}, 
                    Previous reward: {observation.reward}, 
                    Previous reward list: {observation.rewardList}, 
                    Previous reward feedback: {observation.rewardFeedback}, 
                    Step: {i}""".strip(),
            }
        )

        llm_output = client.chat.completions.create(
            model=MODEL,
            messages=MESSAGES,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        action: AppAction = parse_output(llm_output.choices[0].message.content)
        observation: AppObservation = env.step(action)

        HISTORY.append(observation)


if __name__ == "__main__":
    main()

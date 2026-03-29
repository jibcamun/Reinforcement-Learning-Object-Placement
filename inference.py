import os
import re
import base64
import textwrap
from io import BytesIO
from typing import List, Optional, Dict

from openai import OpenAI
import numpy as np
from PIL import Image


API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")

SYSTEM_PROMPT = textwrap.dedent(
    """
    You control a web browser through BrowserGym.
    Reply with exactly one action string.
    The action must be a valid BrowserGym command such as:
    - noop()
    - click('<BID>')
    - type('selector', 'text to enter')
    - fill('selector', 'text to enter')
    - send_keys('Enter')
    - scroll('down')
    Use single quotes around string arguments.
    When clicking, use the BrowserGym element IDs (BIDs) listed in the user message.
    If you are unsure, respond with noop().
    Do not include explanations or additional text.
    """
).strip()


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


if __name__ == "__main__":
    main()

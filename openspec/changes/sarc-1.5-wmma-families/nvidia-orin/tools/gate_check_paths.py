from pathlib import Path
import os
STAGE = Path(os.environ["WMMA_CONFIG"]).parent
PROMPTS = {name: (tag, tokens, str(STAGE / name)) for name, tag, tokens in [("prompt_2048.txt", "prefill", "2048"), ("prompt_real_2048.txt", "real", "2048"), ("prompt_check.txt", "check", "1972"), ("r1304.txt", "r1304", "1792")]}

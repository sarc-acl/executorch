"""The prompts of the next-token checks: prompt -> (log tag, expected prompt tokens, tracked file)."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "../../sarc-1.5-e2e-benchmark/kit/prompts"))
PROMPTS = {"prompt_2048.txt": ("prefill", "2048", os.path.join(KIT, "prompt_2048.txt")),
           "prompt_real_2048.txt": ("real", "2048", os.path.join(KIT, "prompt_real_2048.txt")),
           "prompt_check.txt": ("check", "1972", os.path.join(KIT, "prompt_check.txt")),
           "r1304.txt": ("r1304", "1792", os.path.join(HERE, "r1304.txt"))}

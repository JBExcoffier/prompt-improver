#!/usr/bin/env python3
"""
Script to list all available prompts from RESULT_PATH subfolders.
"""

import os
import json
import sys
import dotenv


dotenv.load_dotenv(dotenv_path="./.env")

RESULT_PATH = os.getenv("RESULT_PATH")


def get_prompt_info(folder_path):
    """Extract prompt information from infos/infos.json file."""
    infos_path = os.path.join(folder_path, "infos", "infos.json")

    try:
        with open(infos_path, "r") as f:
            data = json.load(f)
            return data.get("prompt", "No prompt field found")
    except FileNotFoundError:
        return "No infos.json file found"
    except json.JSONDecodeError:
        return "Invalid JSON in infos.json"
    except Exception as e:
        return f"Error reading file: {e}"


def main():
    """Main function to list all prompts."""
    if not RESULT_PATH:
        print("❌ RESULT_PATH environment variable not found")
        print("Please set RESULT_PATH in your .env file")
        sys.exit(1)

    if not os.path.exists(RESULT_PATH):
        print(f"❌ RESULT_PATH does not exist : {RESULT_PATH}")
        sys.exit(1)

    print(f"Scanning RESULT_PATH : {RESULT_PATH}", "\n")

    subfolders = []
    try:
        for item in os.listdir(RESULT_PATH):
            item_path = os.path.join(RESULT_PATH, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
    except PermissionError:
        print(f"❌ Permission denied accessing : {RESULT_PATH}")
        sys.exit(1)

    if not subfolders:
        print("❌ No subfolders found in RESULT_PATH")
        sys.exit(1)

    subfolders.sort()

    plural = "s" if len(subfolders) > 1 else ""
    print(f"Found {len(subfolders)} subfolder{plural} :\n")

    for folder_name in subfolders:
        folder_path = os.path.join(RESULT_PATH, folder_name)
        prompt = get_prompt_info(folder_path)

        print(f"\tFolder name : {folder_name}")
        print(f"\tPrompt : {prompt}")
        print()


if __name__ == "__main__":
    main()

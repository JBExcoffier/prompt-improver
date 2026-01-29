import dspy
import os
import yaml
import dotenv

from models.languagemodels.openai import OpenAImodel
from models.languagemodels.vllm import VLLMmodel


def load_model_config():
    """Load model configuration from YAML file."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "models.yaml")

    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Validate that model names are unique globally
        external_models = set(config.get("EXTERNAL_API_MODELS", []))
        local_models = set(config.get("LOCAL_MODELS", []))

        # Check for duplicates
        duplicates = external_models.intersection(local_models)
        if duplicates:
            raise ValueError(
                f"Duplicate model names found: {duplicates}. Model names must be unique globally."
            )

        return config

    except FileNotFoundError:
        raise FileNotFoundError(f"Model configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


# Load model configuration
MODEL_CONFIG = load_model_config()
EXTERNAL_API_MODELS = MODEL_CONFIG["EXTERNAL_API_MODELS"]
LOCAL_MODELS = MODEL_CONFIG["LOCAL_MODELS"]
AVAILABLE_MODELS = EXTERNAL_API_MODELS + LOCAL_MODELS


dotenv.load_dotenv(dotenv_path="./.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")
LOCAL_BASE_URL = os.getenv("LOCAL_BASE_URL")


def load_dspy_models(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model name '{model_name}' is not available.")

    if model_name in EXTERNAL_API_MODELS:
        api_key = OPENAI_API_KEY
        base_url = None
    elif model_name in LOCAL_MODELS:
        api_key = LOCAL_API_KEY
        base_url = LOCAL_BASE_URL
        model_name = (
            "hosted_vllm" + "/" + model_name
        )  # See https://docs.litellm.ai/docs/providers/vllm
    else:
        raise ValueError(f"Model name '{model_name}' is not available.")

    return dspy.LM(model_name, api_key=api_key, api_base=base_url)


def get_raw_model(model_name: str):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Model name '{model_name}' is not available.")

    if model_name in EXTERNAL_API_MODELS:
        model = OpenAImodel(model_name=model_name, api_key=OPENAI_API_KEY)
    elif model_name in LOCAL_MODELS:
        model = VLLMmodel(
            model_name=model_name, api_key=LOCAL_API_KEY, base_url=LOCAL_BASE_URL
        )
    else:
        raise ValueError(f"Model name '{model_name}' is not available.")

    return model

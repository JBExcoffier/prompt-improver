import sys

try:
    from models.load import (
        EXTERNAL_API_MODELS,
        LOCAL_MODELS,
        AVAILABLE_MODELS,
        MODEL_CONFIG,
    )

    print("✅ Model configuration loaded successfully", "\n")
    print(f"\tExternal API Models: {EXTERNAL_API_MODELS}")
    print(f"\tLocal Models: {LOCAL_MODELS}")
    print()
    print(f"\tFull Config: {MODEL_CONFIG}", "\n")

    # Uniqueness
    external_set = set(EXTERNAL_API_MODELS)
    local_set = set(LOCAL_MODELS)
    all_set = set(AVAILABLE_MODELS)

    if len(all_set) == len(EXTERNAL_API_MODELS) + len(LOCAL_MODELS):
        print("✅ All model names are unique")
    else:
        print("❌ Duplicate model names found")

except Exception as e:
    print(f"❌ Error loading model configuration: {e}")
    sys.exit(1)

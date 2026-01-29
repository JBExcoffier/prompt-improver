import os
import json
import sys
import dotenv


dotenv.load_dotenv(dotenv_path="./.env")

RESULT_PATH = os.getenv("RESULT_PATH")


def get_dspy_metrics(folder_path):
    """Extract DSPy metrics from metrics/dspy/*.json files."""
    metrics_path = os.path.join(folder_path, "metrics", "dspy")

    if not os.path.exists(metrics_path):
        return "No metrics/dspy folder found"

    metrics_files = []
    try:
        for file in os.listdir(metrics_path):
            if file.endswith(".json"):
                model_name = file.replace(".json", "")
                file_path = os.path.join(metrics_path, file)

                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    binary_accuracy = data.get("binary_accuracy_metrics", None)
                    mape = data.get("mape_metrics", None)

                    metrics_files.append(
                        {
                            "model": model_name,
                            "metrics": {
                                "binary_accuracy": binary_accuracy,
                                "mape": mape,
                            },
                        }
                    )

                except json.JSONDecodeError:
                    metrics_files.append(
                        {
                            "model": model_name,
                            "metrics": {
                                "binary_accuracy": "Invalid JSON",
                                "mape": "Invalid JSON",
                            },
                        }
                    )
                except Exception as e:
                    metrics_files.append(
                        {
                            "model": model_name,
                            "metrics": {
                                "binary_accuracy": f"Error: {e}",
                                "mape": f"Error: {e}",
                            },
                        }
                    )

    except PermissionError:
        return "Permission denied accessing metrics folder"
    except Exception as e:
        return f"Error reading metrics: {e}"

    return metrics_files


def main():
    if not RESULT_PATH:
        print("❌ RESULT_PATH environment variable not found")
        print("Please set RESULT_PATH in your .env file")
        sys.exit(1)

    if not os.path.exists(RESULT_PATH):
        print(f"❌ RESULT_PATH does not exist: {RESULT_PATH}")
        sys.exit(1)

    print(f"Scanning global DSPy metrics in RESULT_PATH : {RESULT_PATH}", "\n")

    subfolders = []
    try:
        for item in os.listdir(RESULT_PATH):
            item_path = os.path.join(RESULT_PATH, item)
            if os.path.isdir(item_path):
                subfolders.append(item)
    except PermissionError:
        print(f"❌ Permission denied accessing: {RESULT_PATH}")
        sys.exit(1)

    if not subfolders:
        print("❌ No subfolders found in RESULT_PATH")
        sys.exit(1)

    subfolders.sort()

    plural = "s" if len(subfolders) > 1 else ""
    print(f"Found {len(subfolders)} subfolder{plural} :\n")

    for folder_name in subfolders:
        folder_path = os.path.join(RESULT_PATH, folder_name)
        models_and_metrics_infos = get_dspy_metrics(folder_path)

        print(f"\tFolder name : {folder_name}")

        if isinstance(models_and_metrics_infos, list):
            for infos in models_and_metrics_infos:
                print(f"\t\tModel name = {infos['model']}")
                print(f"\t\t\tBinary Accuracy = {infos['metrics']['binary_accuracy']}%")
                print(f"\t\t\tMAPE = {infos['metrics']['mape']}%")
                print()
        else:
            print(f"\t\t{models_and_metrics_infos}")

        print()


if __name__ == "__main__":
    main()

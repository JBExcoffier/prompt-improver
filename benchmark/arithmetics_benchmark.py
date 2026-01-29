import pandas
import dspy
import re
import os
import dotenv
import json
import argparse
from tqdm import tqdm
from typing import Dict, Optional, Any

from models.supercalculator import SuperCalculator
from models.load import load_dspy_models, get_raw_model
import utils.dspy_templates as dspy_templates
import utils.metrics as metrics


class ArgumentParser:
    """Handles command line argument parsing and validation."""

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """Parse and validate command line arguments."""
        parser = argparse.ArgumentParser(
            description="Run arithmetic expression benchmark with specified prompt and model",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--prompt-name",
            type=str,
            required=True,
            help="Name of the prompt to use for benchmarking (required)",
        )

        parser.add_argument(
            "--model-name",
            type=str,
            help="Name of the model to use for benchmarking (only required when --prompt-name is 'original')",
        )

        args = parser.parse_args()
        ArgumentParser._validate_arguments(args)
        return args

    @staticmethod
    def _validate_arguments(args: argparse.Namespace) -> None:
        """Validate argument combinations."""
        if args.prompt_name == "original":
            if not args.model_name:
                raise ValueError(
                    "--model-name is required when --prompt-name is 'original'"
                )
        else:
            # If prompt-name is not 'original', ignore model-name even if provided
            args.model_name = None


class EnvironmentManager:
    """Manages environment variables and configuration."""

    def __init__(self):
        self._load_environment()
        self._validate_environment()

    def _load_environment(self) -> None:
        """Load environment variables from .env file."""
        dotenv.load_dotenv(dotenv_path="./.env")

        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.local_api_key = os.getenv("LOCAL_API_KEY")
        self.local_base_url = os.getenv("LOCAL_BASE_URL")
        self.result_path = os.getenv("RESULT_PATH")
        self.golden_dataset_path = (
            "./dataset/arithmetic_expressions_golden_dataset.parquet"
        )

    def _validate_environment(self) -> None:
        """Validate that required environment variables are set."""
        if not self.result_path:
            raise ValueError("RESULT_PATH environment variable is not set")

        # Create result directory if it doesn't exist
        os.makedirs(self.result_path, exist_ok=True)
        print(f"Results will be saved in RESULT_PATH={self.result_path}")


class PromptManager:
    """Manages prompt loading and template creation."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager
        self.template_input = "{INPUT}"

    def load_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """Load prompt configuration based on prompt name."""
        self._validate_prompt_name(prompt_name)

        if prompt_name == "original":
            return self._create_original_prompt()
        else:
            return self._load_optimized_prompt(prompt_name)

    def _validate_prompt_name(self, prompt_name: str) -> None:
        """Validate that the prompt name is available."""
        available_prompt_names = ["original"] + [
            n
            for n in os.listdir(self.env_manager.result_path)
            if os.path.isdir(os.path.join(self.env_manager.result_path, n))
        ]

        if prompt_name not in available_prompt_names:
            raise ValueError(
                f"Prompt name '{prompt_name}' is not available in RESULT_PATH "
                f"({self.env_manager.result_path})."
            )

    def _create_original_prompt(self) -> Dict[str, Any]:
        """Create original prompt configuration."""
        instruction = "Answer"
        prompt = {"name": "original", "prompt": instruction}

        supercalculator = SuperCalculator(instruction=instruction)
        template_messages = dspy.ChatAdapter().format(
            signature=supercalculator.signature,
            demos=[],
            inputs=dict(arithmetic_expression=self.template_input),
        )

        prompt["template-messages"] = template_messages
        return prompt

    def _load_optimized_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """Load optimized prompt from JSON file."""
        prompt_path = os.path.join(self.env_manager.result_path, prompt_name)
        infos_path = os.path.join(prompt_path, "infos", "infos.json")

        with open(infos_path, "r") as json_file:
            return json.load(json_file)


class ModelManager:
    """Manages model loading and configuration."""

    @staticmethod
    def get_model_name(prompt_name: str, provided_model_name: Optional[str]) -> str:
        """Determine the model name based on prompt name."""
        if prompt_name == "original":
            if not provided_model_name:
                raise ValueError(
                    "MODEL_NAME is required when PROMPT_NAME is 'original'"
                )
            return provided_model_name
        else:
            # Extract model name from prompt name pattern: student-MODEL-teacher-...
            match = re.search(r"student-(.*?)-teacher", prompt_name)
            if not match:
                raise ValueError(
                    f"MODEL_NAME couldn't be retrieved from PROMPT_NAME '{prompt_name}'. "
                    f"Expected pattern: 'student-MODEL-teacher-TEACHER'"
                )

            model_name = match.group(1)
            print(f"Retrieved MODEL_NAME from prompt: {model_name}")
            return model_name


class BenchmarkRunner:
    """Main benchmark execution class."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager
        self.prompt_manager = PromptManager(env_manager)

    def run_benchmark(self, prompt_name: str, model_name: Optional[str]) -> None:
        """Run the complete benchmark process."""
        # Load prompt and determine model
        prompt = self.prompt_manager.load_prompt(prompt_name)
        final_model_name = ModelManager.get_model_name(prompt_name, model_name)

        # Load dataset
        expressions = self._load_dataset()

        # Create output directories
        output_paths = self._create_output_directories(prompt_name)

        # Save prompt configuration
        self._save_prompt_config(prompt, output_paths["infos"])

        # Run benchmark
        results = self._run_benchmark_execution(expressions, prompt, final_model_name)

        # Save results
        self._save_results(results, final_model_name, output_paths["results"])

        # Calculate and save DSPy metrics
        self._calculate_dspy_metrics(
            expressions, prompt, final_model_name, output_paths["metrics"]
        )

        print("Benchmark completed successfully !")

    def _load_dataset(self) -> pandas.DataFrame:
        """Load the golden dataset."""
        expressions = pandas.read_parquet(
            path=self.env_manager.golden_dataset_path
        ).head(
            20
        )  # TODO : remove .head to run on whole dataset
        print(f"Golden dataset loaded with {len(expressions)} arithmetics expressions.")
        return expressions

    def _create_output_directories(self, prompt_name: str) -> Dict[str, str]:
        """Create necessary output directories."""
        prompt_path = os.path.join(self.env_manager.result_path, prompt_name)

        directories = {
            "base": prompt_path,
            "results": os.path.join(prompt_path, "results"),
            "infos": os.path.join(prompt_path, "infos"),
            "metrics": os.path.join(prompt_path, "metrics", "dspy"),
        }

        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)

        return directories

    def _save_prompt_config(self, prompt: Dict[str, Any], infos_path: str) -> None:
        """Save prompt configuration to JSON file."""
        infos_file = os.path.join(infos_path, "infos.json")
        with open(infos_file, "w") as json_file:
            json.dump(prompt, json_file, indent=4)

    def _run_benchmark_execution(
        self, expressions: pandas.DataFrame, prompt: Dict[str, Any], model_name: str
    ) -> pandas.DataFrame:
        """Execute the benchmark on all expressions."""
        print("Starting benchmark process")

        model = get_raw_model(model_name=model_name)
        results = []

        for _, row in tqdm(
            expressions.iterrows(),
            total=len(expressions),
            desc="Processing expressions",
        ):
            messages = dspy_templates.get_messages_from_dspy_template(
                template_messages=prompt["template-messages"], input=row["expression"]
            )

            process_error = None
            try:
                result = model.get_string_output(messages=messages)
            except Exception as e:
                process_error = str(e)
                result = None

            results.append(
                {"id": row["id"], "process_error": process_error, "raw_result": result}
            )

        return pandas.DataFrame(results)

    def _save_results(
        self, results: pandas.DataFrame, model_name: str, results_path: str
    ) -> None:
        """Save benchmark results to parquet file."""
        results_file = os.path.join(results_path, f"{model_name}.parquet")
        results.to_parquet(path=results_file, index=False)
        print("Raw benchmark results saved.")

    def _calculate_dspy_metrics(
        self,
        expressions: pandas.DataFrame,
        prompt: Dict[str, Any],
        model_name: str,
        metrics_path: str,
    ) -> None:
        """Calculate and save DSPy metrics."""
        print("Calculating global DSPy metrics.")

        # Create training set
        trainset = []
        for _, row in expressions.iterrows():
            example = dspy.Example(
                arithmetic_expression=row["expression"], result=row["result"]
            ).with_inputs("arithmetic_expression")
            trainset.append(example)

        # Setup evaluator
        evaluator = dspy.Evaluate(
            devset=trainset,
            num_threads=1,
            display_progress=True,
            display_table=1,
            provide_traceback=True,
        )

        # Load DSPy model
        lm = load_dspy_models(model_name=model_name)
        dspy.settings.configure(lm=lm)

        # Create supercalculator
        supercalculator = SuperCalculator(instruction=prompt["prompt"])

        # Calculate metrics
        dspy_metrics = {
            "binary_accuracy_metrics": evaluator(
                dspy.Predict(supercalculator.signature),
                metric=metrics.dspy_binary_accuracy_metrics,
            ),
            "mape_metrics": evaluator(
                dspy.Predict(supercalculator.signature),
                metric=metrics.dspy_mape_metrics,
            ),
        }

        # Save metrics
        metrics_file = os.path.join(metrics_path, f"{model_name}.json")
        with open(metrics_file, "w") as json_file:
            json.dump(dspy_metrics, json_file, indent=4)

        print("Global DSPy metrics saved.")


def main():
    """Main entry point for the benchmark script."""
    try:
        print("Loading environment and argument variables.")

        args = ArgumentParser.parse_arguments()
        env_manager = EnvironmentManager()

        print("Environment and argument variables correctly loaded.")

        # Run benchmark
        benchmark_runner = BenchmarkRunner(env_manager)
        benchmark_runner.run_benchmark(args.prompt_name, args.model_name)

    except Exception as e:
        print(f"❌ Error during benchmark execution: {e}")
        raise


if __name__ == "__main__":
    main()

import pandas
import dspy
import json
import os
import dotenv
import argparse
from typing import Dict, Any, Callable

from models.supercalculator import SuperCalculator
from models.load import load_dspy_models
import utils.metrics as metrics


class ArgumentParser:
    """Handles command line argument parsing and validation."""

    @staticmethod
    def parse_arguments() -> argparse.Namespace:
        """Parse and validate command line arguments."""
        parser = argparse.ArgumentParser(
            description="Run DSPy prompt optimization with specified student and teacher models",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "--student-model-name",
            type=str,
            required=True,
            help="Name of the student model to use for optimization (required)",
        )

        parser.add_argument(
            "--teacher-model-name",
            type=str,
            required=True,
            help="Name of the teacher model to use for optimization (required)",
        )

        parser.add_argument(
            "--optimization-metrics",
            type=str,
            choices=["accuracy", "mape"],
            default="mape",
            help="Metrics to use for optimization ('accuracy' or 'mape'). Default is 'mape'",
        )

        return parser.parse_args()


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


class MetricsManager:
    """Manages optimization metrics selection and configuration."""

    @staticmethod
    def get_optimization_function(metrics_name: str) -> Callable:
        """Get the optimization metrics function based on the metrics name."""
        metrics_mapping = {
            "accuracy": metrics.dspy_binary_accuracy_metrics,
            "mape": metrics.dspy_mape_metrics,
        }

        if metrics_name not in metrics_mapping:
            raise ValueError(
                f"OPTIMIZATION_METRICS must be in ['accuracy', 'mape'] but '{metrics_name}' was passed."
            )

        return metrics_mapping[metrics_name]


class DatasetManager:
    """Manages dataset loading and preparation."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager

    def load_dataset(self) -> pandas.DataFrame:
        """Load the golden dataset."""
        print("Loading golden dataset.")
        expressions = pandas.read_parquet(
            path=self.env_manager.golden_dataset_path
        ).head(
            20
        )  # TODO: remove .head to run on whole dataset
        print(f"Golden dataset loaded with {len(expressions)} arithmetic expressions.")
        return expressions

    def create_trainset(self, expressions: pandas.DataFrame) -> list:
        """Create DSPy training set from expressions."""
        trainset = []
        for _, row in expressions.iterrows():
            example = dspy.Example(
                arithmetic_expression=row["expression"], result=row["result"]
            ).with_inputs("arithmetic_expression")
            trainset.append(example)
        return trainset


class PromptManager:
    """Manages prompt loading and template creation."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager
        self.original_prompt_name = "original"
        self.template_input = "{INPUT}"

    def load_original_prompt(self) -> Dict[str, Any]:
        """Load the original prompt configuration."""
        original_prompt_path = os.path.join(
            self.env_manager.result_path,
            self.original_prompt_name,
            "infos",
            "infos.json",
        )

        with open(original_prompt_path, "r") as json_file:
            return json.load(json_file)

    def create_supercalculator(self, prompt: Dict[str, Any]) -> SuperCalculator:
        """Create SuperCalculator instance from prompt."""
        return SuperCalculator(instruction=prompt["prompt"])


class ModelManager:
    """Manages model loading and configuration."""

    @staticmethod
    def setup_models(student_model_name: str, teacher_model_name: str) -> tuple:
        """Setup student and teacher models for optimization."""
        print(f"Setting up student model: {student_model_name}")
        student_lm = load_dspy_models(model_name=student_model_name)
        dspy.settings.configure(lm=student_lm)

        print(f"Setting up teacher model: {teacher_model_name}")
        teacher_lm = load_dspy_models(model_name=teacher_model_name)

        return student_lm, teacher_lm


class OptimizationManager:
    """Manages the DSPy optimization process."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager

    def run_optimization(
        self,
        student_lm: dspy.LM,
        teacher_lm: dspy.LM,
        supercalculator: SuperCalculator,
        trainset: list,
        optimization_function: Callable,
    ) -> dspy.Module:
        """Run DSPy MIPROv2 optimization."""
        print("Starting DSPy prompt optimization")

        tp = dspy.MIPROv2(
            metric=optimization_function,
            auto="light",
            task_model=student_lm,
            prompt_model=teacher_lm,
        )

        optimized_prompt = tp.compile(
            student=dspy.Predict(supercalculator.signature),
            trainset=trainset,
            max_labeled_demos=0,
            max_bootstrapped_demos=0,
        )

        print("DSPy prompt optimization completed.")
        return optimized_prompt


class OutputManager:
    """Manages output directory creation and file saving."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager

    def create_output_directories(self, prompt_name: str) -> Dict[str, str]:
        """Create necessary output directories."""
        prompt_path = os.path.join(self.env_manager.result_path, prompt_name)

        directories = {
            "base": prompt_path,
            "results": os.path.join(prompt_path, "results"),
            "infos": os.path.join(prompt_path, "infos"),
        }

        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)

        return directories

    def save_optimized_prompt(
        self, optimized_prompt: Dict[str, Any], infos_path: str
    ) -> None:
        """Save optimized prompt configuration to JSON file."""
        infos_file = os.path.join(infos_path, "infos.json")
        with open(infos_file, "w") as json_file:
            json.dump(optimized_prompt, json_file, indent=4)
        print("Optimized prompt configuration saved.")


class PromptOptimizer:
    """Main prompt optimization execution class."""

    def __init__(self, env_manager: EnvironmentManager):
        self.env_manager = env_manager
        self.dataset_manager = DatasetManager(env_manager)
        self.prompt_manager = PromptManager(env_manager)
        self.optimization_manager = OptimizationManager(env_manager)
        self.output_manager = OutputManager(env_manager)

    def optimize_prompt(
        self,
        student_model_name: str,
        teacher_model_name: str,
        optimization_metrics: str,
    ) -> None:
        """Run the complete prompt optimization process."""
        print("Starting prompt optimization process")

        optimization_function = MetricsManager.get_optimization_function(
            optimization_metrics
        )

        # Golden dataset
        expressions = self.dataset_manager.load_dataset()
        trainset = self.dataset_manager.create_trainset(expressions)

        # Load original prompt
        original_prompt = self.prompt_manager.load_original_prompt()
        supercalculator = self.prompt_manager.create_supercalculator(original_prompt)

        student_lm, teacher_lm = ModelManager.setup_models(
            student_model_name, teacher_model_name
        )

        # Prompt optimization
        optimized_prompt_module = self.optimization_manager.run_optimization(
            student_lm, teacher_lm, supercalculator, trainset, optimization_function
        )

        optimized_prompt = self._create_optimized_prompt_config(
            optimized_prompt_module, student_model_name, teacher_model_name
        )

        # Save results
        output_paths = self.output_manager.create_output_directories(
            optimized_prompt["name"]
        )
        self.output_manager.save_optimized_prompt(
            optimized_prompt, output_paths["infos"]
        )

        print("Prompt optimization completed successfully!")

    def _create_optimized_prompt_config(
        self,
        optimized_prompt_module: dspy.Module,
        student_model_name: str,
        teacher_model_name: str,
    ) -> Dict[str, Any]:
        """Create optimized prompt configuration from DSPy module."""
        optimized_prompt = {
            "name": f"student-{student_model_name}-teacher-{teacher_model_name}",
            "prompt": optimized_prompt_module.signature.instructions,
        }

        optimized_template_messages = dspy.ChatAdapter().format(
            signature=optimized_prompt_module.signature,
            demos=[],
            inputs=dict(arithmetic_expression="{INPUT}"),
        )

        optimized_prompt["template-messages"] = optimized_template_messages
        return optimized_prompt


def main():
    """Main entry point for the prompt optimization script."""
    try:
        print("Loading environment and argument variables.")

        args = ArgumentParser.parse_arguments()
        env_manager = EnvironmentManager()

        print("Environment and argument variables correctly loaded.")

        # Run prompt optimization
        prompt_optimizer = PromptOptimizer(env_manager)
        prompt_optimizer.optimize_prompt(
            args.student_model_name, args.teacher_model_name, args.optimization_metrics
        )

    except Exception as e:
        print(f"❌ Error during prompt optimization: {e}")
        raise


if __name__ == "__main__":
    main()

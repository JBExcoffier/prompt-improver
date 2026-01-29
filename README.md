# Auto prompt improver using DSPy

👋 Welcome to the repo with all the reproductible code as well as images used for the [_Auto prompt improver using DSPy_ blogpost on Medium](https://medium.com/@jb.excoffier/auto-prompt-improver-using-dspy-c095c9c402ef).




## _About_

This repo contains a benchmark of a prompt improver technique on an arithmetic dataset 🤖

🐍 The code is in Python. It also uses [vLLM](https://docs.vllm.ai/en/latest/) to run language models locally.

📖 The blogost is accessible on Medium here : **https://medium.com/@jb.excoffier/auto-prompt-improver-using-dspy-c095c9c402ef**


## _Folder structure_

⭐ This repo contains the following :
- 💻 All code files for 
  - Toy dataset for benchmark consisting in arithmetic operations in `dataset`.
  - Make the prompt improver using `DSPy` package work for multiple models, including local ones in `auto_prompt_optimization`.
  - Run the benchmark for each model and prompt in `benchmark`.
  - Explore the generated results in `exploration` using CLI or a notebook.
    - You can use the already generated results located in the `results` folder or generate your own results using other language models 😃.
- 📊 All images that were generated and used in the blogpost in the `blogpost` floder.

## _Running it_ 😉

📌 The blogpost _**Auto prompt improver using DSPy**_ explains in details every step along with commands that are needed to make this project run on your own laptop.

First of all you just need to add an `.env` file at the root of the project with the following structure (as specified in the `.env.default` file) : 

```
RESULT_PATH=LOCATION_OF_GENERATED_RESULTS
OPENAI_API_KEY=YOUR_OWN_OPENAI_API_KEY
LOCAL_API_KEY=super-strong-secret-token-for-local-lm
LOCAL_BASE_URL=http://0.0.0.0:8080/v1
```


👏 You can follow the [blogpost](https://medium.com/@jb.excoffier/auto-prompt-improver-using-dspy-c095c9c402ef) to set up and run everything correctly !

⌨️ Still, here is a **quick view of all commands** that need to be executed at the root of the project to run the benchmark and explore the results :

- **Models** :
  - New models should be added to the `config/models.yaml` config file.
  - List available models, both external api and local models, with `python -m exploration.test_model_config`

- **Running the benchmark**
  - Prompt optimizer is `auto_prompt_optimization/dspy_prompt_optimization.py` :
    - Get help with `python -m auto_prompt_optimization.dspy_prompt_optimization --help`
    - Example : `python -m auto_prompt_optimization.dspy_prompt_optimization --student-model-name 'gpt-4.1-nano' --teacher-model-name 'gpt-4.1-nano'`
    - Specify optimization metrics (default is `mape`) : `python -m auto_prompt_optimization.dspy_prompt_optimization --student-model-name 'gpt-4.1-nano' --teacher-model-name 'gpt-4.1' --optimization-metrics 'accuracy'`
  - Benchmark is `benchmark/arithmetics_benchmark.py` :
    - Get help with `python -m benchmark.arithmetics_benchmark --help`
    - Examples :
      - `python -m benchmark.arithmetics_benchmark --prompt-name 'original' --model-name 'gpt-4.1-nano'`
      - Local model (deployed with vLLM) : `python -m benchmark.arithmetics_benchmark --prompt-name 'original' --model-name 'LocalSmolLM135m'`

- **Explore the generated results** (or already existing ones in the `results` folder)
  - List all available prompts with `python -m exploration.list_prompts`
  - List all available global metrics for each prompts and models (global metrics computed using `DSPy`) with `python -m exploration.print_global_metrics`

from dspy import Example


def dspy_binary_accuracy_metrics(example: Example, prediction: Example, trace=None):
    return prediction.result == example.result


def dspy_mape_metrics(example: Example, prediction: Example, trace=None):
    return abs((prediction.result - example.result) / example.result)

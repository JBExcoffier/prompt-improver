import dspy


class SuperCalculator:
    def __init__(self, instruction: str):
        self.instruction = instruction
        self.set_signature()

    def set_signature(self):
        class InnerCalculator(dspy.Signature):
            arithmetic_expression: str = dspy.InputField(
                desc="Arithmetic expression to evaluate."
            )
            result: float = dspy.OutputField(
                desc="Evaluated result from the given arithmetic expression."
            )

        InnerCalculator.instructions = self.instruction

        self.signature = InnerCalculator

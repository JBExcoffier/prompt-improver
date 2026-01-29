import re


def string_result_to_numeric(result: str):
    """
    Convert a string result to a numeric value (float or int).

    Args:
        result (str): The string to convert

    Returns:
        float: The numeric value

    Raises:
        ValueError: If the string cannot be converted to a numeric value
    """
    if not isinstance(result, str):
        raise ValueError(f"Input must be a string, got {type(result)}")

    # Strip whitespace
    result = result.strip()

    # - Optional sign (+ or -)
    # - One or more digits
    # - Optional decimal point followed by one or more digits
    numeric_pattern = r"^[+-]?\d+(?:\.\d+)?$"

    if re.match(numeric_pattern, result):
        try:
            return float(result)
        except ValueError:
            raise ValueError(f"String result cannot be converted to float ({result})")
    raise ValueError(f"String result cannot be converted to float ({result})")

import re


def get_messages_from_dspy_template(
    template_messages: list, input: str, template_input: str = "{INPUT}"
):
    messages = []

    for m in template_messages:
        role = m["role"]
        content = m["content"].replace(template_input, input)

        messages.append({"role": role, "content": content})

    return messages


def get_result_from_dspy_template(result: str, final_strip: bool = True):
    pattern = (
        r"\[\[ ## result ## \]\](?:\n*)\s*(-?[\d.]+)\s*(?:\n*)\[\[ ## completed ## \]\]"
    )

    # Search for the number
    match = re.search(pattern, result)

    # Print the result if found
    if match:
        result = match.group(1)
        if final_strip:
            result = result.strip()
        return result
    else:
        raise ValueError(f"Result cannot be correctly retrieved ({result}).")

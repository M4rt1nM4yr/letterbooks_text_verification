import re

def extract_abbreviations(input_str, start_token, end_token):
    if isinstance(input_str, str):
        input_str = [input_str]
    start_token_escaped = re.escape(start_token)
    end_token_escaped = re.escape(end_token)

    # Regex pattern to match text between start_token and end_token
    pattern = f"{start_token_escaped}(.*?){end_token_escaped}"

    output = []
    for i in input_str:
        # Finding all matches using the pattern
        matches = re.findall(pattern, i)
        output.append(" ".join(matches))

    return output

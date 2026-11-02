import html
import re


def clean_text(text):
    """
    Clean resume or job description text.

    Steps:
    1. Handle missing values.
    2. Convert text to lowercase.
    3. Remove HTML tags.
    4. Remove extra whitespace.
    """

    if not isinstance(text, str):
        return ""

    # Convert text to lowercase
    text = text.lower()
    text = html.unescape(text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()

def get_text_length(text):
    """
    Return the number of characters in the text.
    """
    if not isinstance(text, str):
        return 0

    return len(text)

if __name__ == "__main__":
    sample = "Python &amp; SQL developer with 3 years experience"

    cleaned = clean_text(sample)
    length = get_text_length(cleaned)

    print("Original:", sample)
    print("Cleaned:", cleaned)
    print("Text length:", length)
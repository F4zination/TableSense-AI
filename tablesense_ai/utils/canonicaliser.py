import re
from fractions import Fraction

# Currency symbols to remove
CURRENCY_REGEX = r"[$€£¥]"

# Thousand separators to remove (1,234 -> 1234)
THOUSAND_SEP_REGEX = r"(?<=\d),(?=\d{3}\b)"

def simplify_fraction(text):
    """
    If text is a fraction (e.g., '4/8'), simplify to lowest terms ('1/2').
    Otherwise return None.
    """
    match = re.fullmatch(r"\s*(-?\d+)\s*/\s*(-?\d+)\s*", text)
    if not match:
        return None

    num, den = map(int, match.groups())
    if den == 0:
        return text  # avoid division errors

    frac = Fraction(num, den)
    return f"{frac.numerator}/{frac.denominator}"


def normalize_number(text):
    """
    Normalize number formatting:
    - Remove currency
    - Remove thousand separators
    - Convert decimals
    - Use '.' as decimal separator
    - Round to 2 decimals
    - Remove trailing .00 for integers
    - Remove trailing zeros for decimals (1.20 -> 1.2)
    """

    # 1. remove currency symbols
    cleaned = re.sub(CURRENCY_REGEX, "", text).strip()

    # 2. remove thousand separators
    cleaned = re.sub(THOUSAND_SEP_REGEX, "", cleaned)

    # 3. replace comma-decimal (10,45 -> 10.45)
    cleaned = cleaned.replace(",", ".")

    # 4. is it a valid number?
    if not re.fullmatch(r"-?\d+(\.\d+)?", cleaned):
        return None

    # cast to float
    value = float(cleaned)

    # round to 2 decimals
    rounded = round(value, 2)

    # whole number?
    if rounded.is_integer():
        return str(int(rounded))

    # decimal number: format to 2 places, then strip trailing zeros and dot
    # Example: 1.20 -> 1.2, 1.25 -> 1.25
    return f"{rounded:.2f}".rstrip('0').rstrip('.')


def clean(pred: str) -> str:
    """
    Apply all formatting rules to a prediction string.
    """
    pred = pred.strip()

    # 1. simplify fractions
    simplified = simplify_fraction(pred)
    if simplified is not None:
        return simplified

    # 2. numeric normalization
    normalized = normalize_number(pred)
    if normalized is not None:
        return normalized

    # 3. text answers -> lowercase
    return pred.lower()
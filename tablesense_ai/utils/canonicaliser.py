#!/usr/bin/env python3
"""
Clean an input string:
1. Convert everything to lower-case
2. Remove a leading currency symbol (e.g. $, €, £, ¥, ₹) plus any spaces that follow it
3. Strip out every single-quote character (')
"""

import re

def clean(text: str) -> str:
    # 1️⃣ lower-case
    text = text.lower()

    # 2️⃣ drop a single leading currency sign (and the spaces right after it)
    text = re.sub(r'^\s*[$€£¥₹]\s*', '', text)

    # 3️⃣ delete all single quotes
    text = text.replace("'", '')

    # 4️⃣ remove trailing .00 if present
    if text.endswith('.00'):
        text = text[:-3]
    return text.strip()

if __name__ == "__main__":
    user_input = input("Enter text: ")
    print(clean(user_input))

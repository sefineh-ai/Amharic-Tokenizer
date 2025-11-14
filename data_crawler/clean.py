"""
Utility for cleaning Amharic text files.

This module provides tools for:
- Removing duplicates, empty lines, and short lines.
- Removing English letters, digits, and unwanted ASCII punctuation.
- Keeping Amharic characters and valid punctuation.
- Splitting lines into sentences.
- Saving cleaned Amharic sentences to an output file.
"""

import re


def clean_amharic_file(input_path: str, output_path: str, min_length: int = 15):
    """
    Clean an Amharic text file.

    Args:
        input_path (str): Path to raw text file.
        output_path (str): Path to save cleaned text.
        min_length (int): Minimum sentence length to keep.

    Returns:
        None
    """
    cleaned_lines = []
    seen = set()

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        text = line.strip()

        if not text:
            continue

        # Remove Latin letters, digits, and unwanted ASCII punctuation
        text = re.sub(r"[A-Za-z@#$%^&*_=+{}\[\]|\\<>~,]+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Split by Amharic + modern punctuation
        split_sentences = re.split(r"[።፤፣፥፧?!\"']+", text)

        for s in split_sentences:
            s = s.strip()
            if len(s) >= min_length and s not in seen and re.search(r"[\u1200-\u137F]", s):
                seen.add(s)
                cleaned_lines.append(s)

    # Write cleaned sentences
    with open(output_path, "w", encoding="utf-8") as f:
        for line in cleaned_lines:
            f.write(line + "\n")

    print(f"✅ Cleaned {len(cleaned_lines)} sentences written to {output_path}")


if __name__ == "__main__":
    clean_amharic_file("raw_amharic.txt",
                       "cleaned_raw_amharic.txt", min_length=15)

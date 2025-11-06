#!/usr/bin/env python3
"""
Text cleaning utilities for SONAR-LLM training
Based on FlagEmbedding approach for NIAH evaluation
Reference: https://github.com/FlagOpen/FlagEmbedding/blob/master/research/Long_LLM/activation_beacon/main/eval_needle.py
"""

import re
from typing import Optional


def clean_text_flag_style(text: str) -> str:
    """
    Clean text using FlagEmbedding style for predictions
    
    This is minimal cleaning used in FlagEmbedding eval_needle.py:
    - Strip newlines from start/end
    - Take only first line
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text (first line only)
    """
    return text.strip("\n").split("\n")[0]


def clean_text_ruler_style(text: str) -> str:
    """
    Clean text for RULER benchmark data generation
    More aggressive cleaning for synthetic data
    
    Based on common practices in RULER and FlagEmbedding data preparation:
    - Remove excessive whitespace
    - Normalize line breaks
    - Remove control characters
    - Standardize spacing
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned and normalized text
    """
    # Remove excessive newlines (3+ → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces (2+ → 1)
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove special control characters
    # \x00-\x08: NULL, SOH, STX, ETX, EOT, ENQ, ACK, BEL, BS
    # \x0b: VT (vertical tab)
    # \x0c: FF (form feed)
    # \x0e-\x1f: SO, SI, DLE, DC1-4, NAK, SYN, ETB, CAN, EM, SUB, ESC, FS, GS, RS, US
    # \x7f-\x9f: DEL and C1 control codes
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Convert all types of whitespace to standard space
    # This handles \t, \r, \n, etc.
    text = re.sub(r'\s', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def postprocess_prediction(
    prediction: str,
    style: str = "ruler",
    max_lines: Optional[int] = None
) -> str:
    """
    Postprocess model prediction for evaluation
    
    Args:
        prediction: Raw model output
        style: Cleaning style ('flag' or 'ruler')
        max_lines: Maximum number of lines to keep (None = all)
        
    Returns:
        Cleaned prediction
    """
    if style == "flag":
        # FlagEmbedding style: first line only
        return clean_text_flag_style(prediction)
    
    elif style == "ruler":
        # RULER style: more thorough cleaning
        # Remove non-printable characters
        np_pattern = re.compile(r'[\x00-\x1f]')
        prediction = np_pattern.sub('\n', prediction).strip()
        
        if max_lines is not None:
            lines = prediction.split('\n')
            prediction = '\n'.join(lines[:max_lines])
        
        return prediction
    
    else:
        raise ValueError(f"Unknown style: {style}")


def normalize_answer(answer: str) -> str:
    """
    Normalize answer for comparison
    
    Args:
        answer: Answer text
        
    Returns:
        Normalized answer (lowercase, stripped)
    """
    return answer.lower().strip()


if __name__ == "__main__":
    # Test the cleaning functions
    test_text = """
    
    This is   a test  text  with
    
    
    excessive    whitespace
    and\tsome\ttabs
    and control characters: \x00\x01
    """
    
    print("Original text:")
    print(repr(test_text))
    print("\nCleaned (RULER style):")
    print(repr(clean_text_ruler_style(test_text)))
    print("\nCleaned (Flag style):")
    print(repr(clean_text_flag_style(test_text)))


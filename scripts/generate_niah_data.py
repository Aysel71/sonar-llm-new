#!/usr/bin/env python3
"""
Generate NIAH (Needle-in-a-Haystack) dataset for SONAR-LLM training
Following RULER benchmark specifications
"""

import os
import sys
import random
import argparse
import json
import uuid
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.text_cleaning import clean_text_ruler_style


class NIAHDataGenerator:
    """Generator for NIAH dataset following RULER format"""
    
    NEEDLE_TEMPLATE = "The special magic number for {key} is: {value}."
    
    # Distractor words for generating haystack
    DISTRACTOR_WORDS = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur",
        "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
        "incididunt", "ut", "labore", "et", "dolore", "magna",
        "aliqua", "enim", "ad", "minim", "veniam", "quis",
        "nostrud", "exercitation", "ullamco", "laboris", "nisi",
        "ut", "aliquip", "ex", "ea", "commodo", "consequat",
        "duis", "aute", "irure", "dolor", "in", "reprehenderit",
        "voluptate", "velit", "esse", "cillum", "dolore",
        "fugiat", "nulla", "pariatur", "excepteur", "sint",
        "occaecat", "cupidatat", "non", "proident", "sunt",
        "culpa", "qui", "officia", "deserunt", "mollit", "anim",
        "id", "est", "laborum"
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        random.seed(seed)
        self.seed = seed
    
    def generate_haystack(self, length: int) -> str:
        """
        Generate distractor text (haystack)
        
        Args:
            length: Approximate length in characters
            
        Returns:
            Generated haystack text
        """
        # Calculate number of words (assuming avg 6 chars per word)
        num_words = max(1, int(length / 6))
        
        text_parts = []
        words_added = 0
        
        while words_added < num_words:
            # Generate sentences of 8-15 words
            sentence_length = random.randint(8, 15)
            if words_added + sentence_length > num_words:
                sentence_length = num_words - words_added
            
            sentence = " ".join(random.choice(self.DISTRACTOR_WORDS) 
                              for _ in range(sentence_length))
            sentence = sentence.capitalize() + ". "
            
            text_parts.append(sentence)
            words_added += sentence_length
        
        return "".join(text_parts).strip()
    
    def generate_needle(
        self, 
        needle_type_k: str = "words",
        needle_type_v: str = "numbers"
    ) -> Tuple[str, str]:
        """
        Generate needle (key-value pair)
        
        Args:
            needle_type_k: Type of key ('words', 'numbers', 'uuids')
            needle_type_v: Type of value ('words', 'numbers', 'uuids')
            
        Returns:
            Tuple of (key, value)
        """
        # Generate key
        if needle_type_k == "words":
            adjectives = ["red", "blue", "green", "yellow", "purple", "orange"]
            nouns = ["apple", "book", "car", "dog", "tree", "house"]
            key = f"{random.choice(adjectives)}_{random.choice(nouns)}"
        elif needle_type_k == "numbers":
            key = f"key_{random.randint(1000, 9999)}"
        elif needle_type_k == "uuids":
            key = f"key_{str(uuid.uuid4())[:8]}"
        else:
            raise ValueError(f"Unknown needle_type_k: {needle_type_k}")
        
        # Generate value
        if needle_type_v == "words":
            adjectives = ["happy", "sad", "big", "small", "fast", "slow"]
            nouns = ["cat", "moon", "sun", "star", "cloud", "river"]
            value = f"{random.choice(adjectives)} {random.choice(nouns)}"
        elif needle_type_v == "numbers":
            value = str(random.randint(1000000, 9999999))
        elif needle_type_v == "uuids":
            value = str(uuid.uuid4())
        else:
            raise ValueError(f"Unknown needle_type_v: {needle_type_v}")
        
        return key, value
    
    def generate_sample(
        self,
        context_length: int = 512,
        needle_type_k: str = "words",
        needle_type_v: str = "numbers",
        needle_depth: float = 50.0
    ) -> Dict[str, Any]:
        """
        Generate a single NIAH sample
        
        Args:
            context_length: Target context length in characters
            needle_type_k: Type of needle key
            needle_type_v: Type of needle value  
            needle_depth: Where to insert needle (0-100%)
            
        Returns:
            Dictionary with NIAH sample
        """
        # Generate needle
        key, value = self.generate_needle(needle_type_k, needle_type_v)
        needle_text = self.NEEDLE_TEMPLATE.format(key=key, value=value)
        
        # Generate haystack
        haystack_length = context_length - len(needle_text) - 50  # Leave some margin
        haystack = self.generate_haystack(max(100, haystack_length))
        
        # Insert needle at specified depth
        insertion_pos = int(len(haystack) * (needle_depth / 100.0))
        insertion_pos = max(0, min(insertion_pos, len(haystack) - 1))
        
        context = (
            haystack[:insertion_pos] + 
            " " + needle_text + " " + 
            haystack[insertion_pos:]
        )
        
        # Clean the context
        context = clean_text_ruler_style(context)
        
        # Create prompt and completion
        prompt = (
            f"Read the following text carefully and find the value associated with {key}:\n\n"
            f"{context}\n\n"
            f"What is the value for {key}?"
        )
        
        completion = f"The value for {key} is: {value}"
        
        return {
            "input": prompt,
            "output": completion,
            "key": key,
            "value": value,
            "needle": needle_text,
            "needle_depth": needle_depth,
            "context_length": len(context),
            "needle_type_k": needle_type_k,
            "needle_type_v": needle_type_v
        }
    
    def generate_dataset(
        self,
        num_samples: int,
        context_length: int = 512,
        needle_type_k: str = "words",
        needle_type_v: str = "numbers",
        depth_distribution: str = "uniform"
    ) -> List[Dict[str, Any]]:
        """
        Generate full NIAH dataset
        
        Args:
            num_samples: Number of samples to generate
            context_length: Target context length
            needle_type_k: Type of needle keys
            needle_type_v: Type of needle values
            depth_distribution: How to distribute needle depths
                               ('uniform', 'random', 'fixed')
            
        Returns:
            List of NIAH samples
        """
        samples = []
        
        for i in tqdm(range(num_samples), desc="Generating NIAH samples"):
            # Determine needle depth
            if depth_distribution == "uniform":
                # Distribute evenly across 0-100%
                depth = (i % 10) * 10.0
            elif depth_distribution == "random":
                depth = random.uniform(0, 100)
            else:  # fixed
                depth = 50.0
            
            sample = self.generate_sample(
                context_length=context_length,
                needle_type_k=needle_type_k,
                needle_type_v=needle_type_v,
                needle_depth=depth
            )
            samples.append(sample)
        
        return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate NIAH dataset for SONAR-LLM training"
    )
    
    # Dataset parameters
    parser.add_argument(
        "--num_samples", type=int, default=1000,
        help="Number of samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--context_length", type=int, default=512,
        help="Target context length in characters (default: 512)"
    )
    parser.add_argument(
        "--needle_type_k", type=str, default="words",
        choices=["words", "numbers", "uuids"],
        help="Type of needle keys (default: words)"
    )
    parser.add_argument(
        "--needle_type_v", type=str, default="numbers",
        choices=["words", "numbers", "uuids"],
        help="Type of needle values (default: numbers)"
    )
    parser.add_argument(
        "--depth_distribution", type=str, default="uniform",
        choices=["uniform", "random", "fixed"],
        help="How to distribute needle depths (default: uniform)"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="../data/niah",
        help="Output directory (default: ../data/niah)"
    )
    parser.add_argument(
        "--output_name", type=str, default="niah_dataset.json",
        help="Output filename (default: niah_dataset.json)"
    )
    
    # Random seed
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate dataset
    print(f"\n{'='*80}")
    print("NIAH Dataset Generation")
    print(f"{'='*80}")
    print(f"Samples: {args.num_samples}")
    print(f"Context length: {args.context_length} chars")
    print(f"Needle type (key): {args.needle_type_k}")
    print(f"Needle type (value): {args.needle_type_v}")
    print(f"Depth distribution: {args.depth_distribution}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")
    
    generator = NIAHDataGenerator(seed=args.seed)
    samples = generator.generate_dataset(
        num_samples=args.num_samples,
        context_length=args.context_length,
        needle_type_k=args.needle_type_k,
        needle_type_v=args.needle_type_v,
        depth_distribution=args.depth_distribution
    )
    
    # Save dataset
    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Statistics
    avg_context = sum(s['context_length'] for s in samples) / len(samples)
    
    print(f"\n{'='*80}")
    print("Generation Complete")
    print(f"{'='*80}")
    print(f"✓ Generated {len(samples)} samples")
    print(f"✓ Saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Average context length: {avg_context:.1f} chars")
    print(f"  Target context length: {args.context_length} chars")
    print(f"{'='*80}\n")
    
    # Show examples
    print("Example samples:")
    print("-"*80)
    for i, sample in enumerate(samples[:2]):
        print(f"\nSample {i+1}:")
        print(f"  Key: {sample['key']}")
        print(f"  Value: {sample['value']}")
        print(f"  Needle: {sample['needle']}")
        print(f"  Depth: {sample['needle_depth']}%")
        print(f"  Context (first 150 chars): {sample['input'][:150]}...")
        print("-"*80)


if __name__ == "__main__":
    main()


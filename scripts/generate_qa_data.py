#!/usr/bin/env python3
"""
Generate QA dataset for SONAR-LLM training
Following RULER benchmark QA task specifications
"""

import os
import sys
import random
import argparse
import json
from typing import List, Dict, Any
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.text_cleaning import clean_text_ruler_style


class QADataGenerator:
    """Generator for QA dataset following RULER format"""
    
    # RULER-style QA template
    QA_TEMPLATE = (
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\n"
        "The following are given documents.\n\n"
        "{context}\n\n"
        "Answer the question based on the given documents. "
        "Only give me the answer and do not output any other words.\n\n"
        "Question: {question}\nAnswer:"
    )
    
    # Base QA pairs (SQuAD-style)
    BASE_QA_PAIRS = [
        {
            "context": "The Amazon rainforest is the largest tropical rainforest in the world, covering approximately 5.5 million square kilometers. It is home to an estimated 10% of all species on Earth.",
            "question": "What percentage of Earth's species live in the Amazon rainforest?",
            "answer": "10%",
            "category": "percentage"
        },
        {
            "context": "Python was created by Guido van Rossum and first released in 1991. It emphasizes code readability with significant use of whitespace.",
            "question": "Who created Python?",
            "answer": "Guido van Rossum",
            "category": "person"
        },
        {
            "context": "The Great Wall of China is approximately 21,196 kilometers long. Construction began in the 7th century BC and continued for centuries.",
            "question": "How long is the Great Wall of China?",
            "answer": "21,196 kilometers",
            "category": "measurement"
        },
        {
            "context": "Water boils at 100 degrees Celsius at sea level. This temperature decreases at higher altitudes due to lower atmospheric pressure.",
            "question": "At what temperature does water boil at sea level?",
            "answer": "100 degrees Celsius",
            "category": "temperature"
        },
        {
            "context": "The human brain contains approximately 86 billion neurons. These neurons communicate through electrical and chemical signals.",
            "question": "How many neurons are in the human brain?",
            "answer": "86 billion",
            "category": "number"
        },
        {
            "context": "Mount Everest is 8,849 meters above sea level, making it the highest mountain on Earth. It is located in the Himalayas on the border between Nepal and Tibet.",
            "question": "What is the height of Mount Everest?",
            "answer": "8,849 meters",
            "category": "measurement"
        },
        {
            "context": "The speed of light in vacuum is approximately 299,792,458 meters per second. This is one of the fundamental constants in physics.",
            "question": "What is the speed of light?",
            "answer": "299,792,458 meters per second",
            "category": "measurement"
        },
        {
            "context": "Shakespeare wrote Romeo and Juliet around 1594-1595. It is one of his most famous tragedies and tells the story of two star-crossed lovers.",
            "question": "Who wrote Romeo and Juliet?",
            "answer": "Shakespeare",
            "category": "person"
        },
    ]
    
    # Distractor sentences
    DISTRACTOR_SENTENCES = [
        "Recent studies have shown significant progress in this field.",
        "Researchers continue to investigate various aspects of this phenomenon.",
        "The data suggests multiple interpretations are possible.",
        "Further analysis reveals interesting patterns in the results.",
        "Evidence indicates a strong correlation between these factors.",
        "Scientists have proposed several theories to explain this.",
        "The findings contribute to our understanding of the subject.",
        "Ongoing research explores different methodologies.",
    ]
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed"""
        random.seed(seed)
        self.seed = seed
    
    def generate_distractor_context(self, length: int) -> str:
        """Generate distractor sentences"""
        text_parts = []
        current_length = 0
        
        while current_length < length:
            sentence = random.choice(self.DISTRACTOR_SENTENCES)
            text_parts.append(sentence)
            current_length += len(sentence) + 1
        
        return " ".join(text_parts)
    
    def generate_sample(
        self,
        context_length: int = 512,
        add_distractors: bool = True
    ) -> Dict[str, Any]:
        """
        Generate a single QA sample
        
        Args:
            context_length: Target context length
            add_distractors: Whether to add distractor paragraphs
            
        Returns:
            Dictionary with QA sample
        """
        # Select random QA pair
        qa = random.choice(self.BASE_QA_PAIRS)
        
        context = qa["context"]
        question = qa["question"]
        answer = qa["answer"]
        category = qa["category"]
        
        # Add distractors if needed
        if add_distractors:
            current_length = len(context)
            distractor_needed = max(0, context_length - current_length - 100)
            
            if distractor_needed > 0:
                # Add 1-3 distractor paragraphs
                num_distractors = random.randint(1, 3)
                distractors = []
                
                for _ in range(num_distractors):
                    dist_text = self.generate_distractor_context(
                        distractor_needed // num_distractors
                    )
                    distractors.append(dist_text)
                
                # Mix real context with distractors
                all_paragraphs = distractors + [context]
                random.shuffle(all_paragraphs)
                full_context = "\n\n".join(all_paragraphs)
            else:
                full_context = context
        else:
            full_context = context
        
        # Clean context
        full_context = clean_text_ruler_style(full_context)
        
        # Create prompt using RULER template
        prompt = self.QA_TEMPLATE.format(
            context=full_context,
            question=question
        )
        
        return {
            "input": prompt,
            "output": answer,
            "question": question,
            "answer": answer,
            "category": category,
            "context_length": len(full_context)
        }
    
    def generate_dataset(
        self,
        num_samples: int,
        context_length: int = 512,
        add_distractors: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate full QA dataset
        
        Args:
            num_samples: Number of samples to generate
            context_length: Target context length
            add_distractors: Whether to add distractor text
            
        Returns:
            List of QA samples
        """
        samples = []
        
        for i in tqdm(range(num_samples), desc="Generating QA samples"):
            sample = self.generate_sample(
                context_length=context_length,
                add_distractors=add_distractors
            )
            samples.append(sample)
        
        return samples


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA dataset for SONAR-LLM training"
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
        "--add_distractors", action="store_true",
        help="Add distractor paragraphs"
    )
    
    # Output parameters
    parser.add_argument(
        "--output_dir", type=str, default="../data/qa",
        help="Output directory (default: ../data/qa)"
    )
    parser.add_argument(
        "--output_name", type=str, default="qa_dataset.json",
        help="Output filename (default: qa_dataset.json)"
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
    print("QA Dataset Generation")
    print(f"{'='*80}")
    print(f"Samples: {args.num_samples}")
    print(f"Context length: {args.context_length} chars")
    print(f"Add distractors: {args.add_distractors}")
    print(f"Random seed: {args.seed}")
    print(f"{'='*80}\n")
    
    generator = QADataGenerator(seed=args.seed)
    samples = generator.generate_dataset(
        num_samples=args.num_samples,
        context_length=args.context_length,
        add_distractors=args.add_distractors
    )
    
    # Save dataset
    output_path = os.path.join(args.output_dir, args.output_name)
    with open(output_path, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Statistics
    avg_context = sum(s['context_length'] for s in samples) / len(samples)
    categories = {}
    for s in samples:
        cat = s['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n{'='*80}")
    print("Generation Complete")
    print(f"{'='*80}")
    print(f"✓ Generated {len(samples)} samples")
    print(f"✓ Saved to: {output_path}")
    print(f"\nStatistics:")
    print(f"  Average context length: {avg_context:.1f} chars")
    print(f"  Categories distribution:")
    for cat, count in sorted(categories.items()):
        print(f"    {cat}: {count} ({100*count/len(samples):.1f}%)")
    print(f"{'='*80}\n")
    
    # Show examples
    print("Example samples:")
    print("-"*80)
    for i, sample in enumerate(samples[:2]):
        print(f"\nSample {i+1}:")
        print(f"  Question: {sample['question']}")
        print(f"  Answer: {sample['answer']}")
        print(f"  Category: {sample['category']}")
        print(f"  Context preview: {sample['input'][:200]}...")
        print("-"*80)


if __name__ == "__main__":
    main()


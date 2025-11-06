#!/usr/bin/env python3
"""
Training script for SONAR-LLM on NIAH task
Professional implementation with RULER metrics
"""

import os
import sys
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline
)
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM
)
from transformers.optimization import get_cosine_schedule_with_warmup

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.text_cleaning import clean_text_flag_style


# ============================================================================
# Configuration
# ============================================================================

class TrainingConfig:
    """Training configuration"""
    
    def __init__(self, **kwargs):
        # Data
        self.data_path = kwargs.get('data_path', '../data/niah/niah_dataset.json')
        self.val_split = kwargs.get('val_split', 0.1)
        
        # Model
        self.model_name = kwargs.get('model_name', 'raxtemur/sonar-llm-1.3b')
        self.hidden_size = kwargs.get('hidden_size', 1024)
        self.num_layers = kwargs.get('num_layers', 10)
        self.num_heads = kwargs.get('num_heads', 16)
        
        # Training
        self.epochs = kwargs.get('epochs', 3)
        self.batch_size = kwargs.get('batch_size', 1)
        self.learning_rate = kwargs.get('learning_rate', 2e-5)
        self.weight_decay = kwargs.get('weight_decay', 1e-3)
        self.warmup_steps = kwargs.get('warmup_steps', 100)
        self.grad_accum_steps = kwargs.get('grad_accum_steps', 4)
        
        # Precision
        self.use_fp16 = kwargs.get('use_fp16', False)
        self.use_bf16 = kwargs.get('use_bf16', True)
        
        # Logging
        self.logging_steps = kwargs.get('logging_steps', 10)
        self.eval_steps = kwargs.get('eval_steps', 50)
        self.save_steps = kwargs.get('save_steps', 100)
        
        # Output
        self.output_dir = kwargs.get('output_dir', '../models/niah_model')
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_id = kwargs.get('gpu_id', 0)
    
    @classmethod
    def from_args(cls, args):
        """Create config from argparse arguments"""
        return cls(**vars(args))


# ============================================================================
# Model Components
# ============================================================================

class Projector(nn.Module):
    """Simple linear projector"""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    
    def forward(self, x):
        return self.linear(x)


class NIAHModel(nn.Module):
    """NIAH model wrapper with SONAR encoders"""
    
    def __init__(
        self,
        llama_model: nn.Module,
        forward_proj: nn.Module,
        reverse_proj: nn.Module,
        sonar_decoder: nn.Module
    ):
        super().__init__()
        self.llama_model = llama_model
        self.forward_proj = forward_proj
        self.reverse_proj = reverse_proj
        self.sonar_decoder = sonar_decoder
        
        # Freeze SONAR decoder
        for p in self.sonar_decoder.parameters():
            p.requires_grad = False
    
    def forward(
        self,
        input_embeddings,
        input_texts,
        input_lens,
        output_embeddings,
        output_texts,
        output_lens
    ):
        """
        Forward pass with loss computation
        
        Returns:
            Cross-entropy loss
        """
        device = input_embeddings.device
        B, T_in, _ = input_embeddings.shape
        
        # Project SONAR embeddings to LLaMA space
        input_embs_proj = self.forward_proj(input_embeddings)
        
        # Get LLaMA hidden states
        llama_out = self.llama_model(
            inputs_embeds=input_embs_proj,
            output_hidden_states=True
        )
        last_hidden = llama_out.hidden_states[-1]
        
        # Collect last hidden state for each sequence
        pred_hidden_list = []
        ref_texts_list = []
        
        for b in range(B):
            input_len = input_lens[b]
            if input_len > 0:
                pred_hidden_list.append(last_hidden[b, input_len - 1, :])
                if output_lens[b] > 0:
                    ref_texts_list.append(output_texts[b][0])
                else:
                    ref_texts_list.append("")
        
        if len(pred_hidden_list) == 0:
            return torch.tensor(0.0, device=device)
        
        # Project back to SONAR space
        pred_hidden = torch.stack(pred_hidden_list, dim=0)
        pred_emb_1024 = self.reverse_proj(pred_hidden)
        
        # Get target tokens using SONAR decoder
        with torch.no_grad():
            target_encoder = self.sonar_decoder.tokenizer.create_encoder(
                task="translation",
                lang="eng_Latn",
                mode="target",
                device=device
            )
        
        encoded_texts = [target_encoder(t) for t in ref_texts_list]
        lengths = [et.size(0) for et in encoded_texts]
        max_len = min(max(lengths) if lengths else 0, 256)
        
        if max_len == 0:
            return torch.tensor(0.0, device=device)
        
        # Prepare decoder inputs
        pad_idx = self.sonar_decoder.tokenizer.vocab_info.pad_idx
        dec_ids = torch.full((len(encoded_texts), max_len), pad_idx, dtype=torch.long, device=device)
        labels = torch.full((len(encoded_texts), max_len), pad_idx, dtype=torch.long, device=device)
        
        for i, et in enumerate(encoded_texts):
            dec_ids[i, :min(len(et), max_len)] = et[:max_len]
            et_shifted = torch.cat([et[1:], torch.tensor([3]).to(device)])
            labels[i, :min(len(et_shifted), max_len)] = et_shifted[:max_len]
        
        # Run decoder
        enc_output = pred_emb_1024.unsqueeze(1)
        dec_out, dec_pad_mask = self.sonar_decoder.model.decode(
            seqs=dec_ids,
            padding_mask=None,
            encoder_output=enc_output,
            encoder_padding_mask=None
        )
        final_out = self.sonar_decoder.model.project(dec_out, dec_pad_mask)
        logits = final_out.logits
        
        # Calculate loss
        vocab_size = logits.size(-1)
        logits_2d = logits.view(-1, vocab_size)
        labels_1d = labels.view(-1)
        
        ce_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")
        loss = ce_fn(logits_2d, labels_1d)
        
        return loss
    
    @torch.no_grad()
    def generate(self, input_embeddings, max_output_len=50):
        """Generate predictions"""
        device = input_embeddings.device
        B = input_embeddings.size(0)
        
        # Project to LLaMA space
        input_embs_proj = self.forward_proj(input_embeddings)
        
        # Get LLaMA output
        llama_out = self.llama_model(
            inputs_embeds=input_embs_proj,
            output_hidden_states=True
        )
        last_hidden = llama_out.hidden_states[-1]
        last_position_hidden = last_hidden[:, -1, :]
        
        # Project to SONAR space
        predicted_embeddings = self.reverse_proj(last_position_hidden)
        
        # Generate text
        outputs = []
        for i in range(B):
            emb = predicted_embeddings[i].unsqueeze(0)
            generated_text = self.sonar_decoder.predict(
                emb,
                target_lang="eng_Latn",
                max_seq_len=max_output_len
            )[0]
            outputs.append(generated_text)
        
        return outputs


# ============================================================================
# Dataset
# ============================================================================

class NIAHDataset(Dataset):
    """NIAH dataset with precomputed embeddings"""
    
    def __init__(
        self,
        samples: List[Dict[str, Any]],
        embedder: TextToEmbeddingModelPipeline,
        precompute: bool = True
    ):
        self.samples = samples
        self.embedder = embedder.eval()
        self.cache = {}
        
        if precompute:
            print(f"Precomputing embeddings for {len(samples)} samples...")
            for idx in tqdm(range(len(samples)), desc="Embedding"):
                self._compute_item(idx)
            print("✓ Embeddings precomputed")
    
    def __len__(self):
        return len(self.samples)
    
    def _compute_item(self, idx):
        """Compute embeddings for one sample"""
        if idx in self.cache:
            return self.cache[idx]
        
        sample = self.samples[idx]
        input_text = sample["input"]
        output_text = sample["output"]
        
        # Tokenize into sentences
        try:
            input_sents = sent_tokenize(input_text)
        except:
            input_sents = [input_text]
        
        try:
            output_sents = sent_tokenize(output_text)
        except:
            output_sents = [output_text]
        
        all_sents = input_sents + output_sents
        
        # Get SONAR embeddings
        with torch.no_grad():
            embeddings = self.embedder.predict(all_sents, source_lang="eng_Latn")
        
        input_embs = embeddings[:len(input_sents)]
        output_embs = embeddings[len(input_sents):]
        
        item = {
            "input_embeddings": [emb.cpu() for emb in input_embs],
            "output_embeddings": [emb.cpu() for emb in output_embs],
            "input_texts": input_sents,
            "output_texts": output_sents,
            "key": sample.get("key", ""),
            "value": sample.get("value", sample.get("answer", ""))
        }
        
        self.cache[idx] = item
        return item
    
    def __getitem__(self, idx):
        return self._compute_item(idx)


def collate_fn(batch):
    """Collate function for DataLoader"""
    max_input_len = max(len(item["input_embeddings"]) for item in batch)
    max_output_len = max(len(item["output_embeddings"]) for item in batch)
    B = len(batch)
    
    input_embed_batch = []
    input_text_batch = []
    output_embed_batch = []
    output_text_batch = []
    input_lens = []
    output_lens = []
    keys_batch = []
    values_batch = []
    
    for item in batch:
        # Process input
        input_embs = item["input_embeddings"]
        input_txts = item["input_texts"]
        input_len = len(input_embs)
        
        padded_input_embs = []
        padded_input_txts = []
        
        for i in range(max_input_len):
            if i < input_len:
                padded_input_embs.append(input_embs[i].unsqueeze(0))
                padded_input_txts.append(input_txts[i])
            else:
                padded_input_embs.append(torch.zeros((1, 1024)))
                padded_input_txts.append("")
        
        padded_input_embs = torch.cat(padded_input_embs, dim=0)
        input_embed_batch.append(padded_input_embs)
        input_text_batch.append(padded_input_txts)
        input_lens.append(input_len)
        
        # Process output
        output_embs = item["output_embeddings"]
        output_txts = item["output_texts"]
        output_len = len(output_embs)
        
        padded_output_embs = []
        padded_output_txts = []
        
        for i in range(max_output_len):
            if i < output_len:
                padded_output_embs.append(output_embs[i].unsqueeze(0))
                padded_output_txts.append(output_txts[i])
            else:
                padded_output_embs.append(torch.zeros((1, 1024)))
                padded_output_txts.append("")
        
        padded_output_embs = torch.cat(padded_output_embs, dim=0)
        output_embed_batch.append(padded_output_embs)
        output_text_batch.append(padded_output_txts)
        output_lens.append(output_len)
        
        # Metadata
        keys_batch.append(item["key"])
        values_batch.append(item["value"])
    
    # Stack batches
    input_embed_batch = torch.stack(input_embed_batch, dim=0)
    output_embed_batch = torch.stack(output_embed_batch, dim=0)
    
    return {
        "input_embeddings": input_embed_batch,
        "input_texts": input_text_batch,
        "input_lens": input_lens,
        "output_embeddings": output_embed_batch,
        "output_texts": output_text_batch,
        "output_lens": output_lens,
        "keys": keys_batch,
        "values": values_batch
    }


# ============================================================================
# RULER Evaluation Metrics
# ============================================================================

def string_match_all(predictions: List[str], references: List[str]) -> float:
    """
    RULER metric for NIAH task
    Checks if ALL expected values are present in predictions
    
    Args:
        predictions: List of predicted texts
        references: List of expected values
        
    Returns:
        Score as percentage (0-100)
    """
    correct = 0
    for pred, ref in zip(predictions, references):
        if ref.lower() in pred.lower():
            correct += 1
    
    score = (correct / len(predictions)) * 100 if predictions else 0
    return round(score, 2)


def evaluate_niah(
    model: NIAHModel,
    dataloader: DataLoader,
    device: torch.device,
    num_examples: int = 5
) -> Dict[str, Any]:
    """
    Evaluate NIAH model
    
    Args:
        model: NIAH model
        dataloader: Validation dataloader
        device: Device to use
        num_examples: Number of examples to return
        
    Returns:
        Dictionary with metrics and examples
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    all_keys = []
    all_values = []
    total_loss = 0.0
    total_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_embeddings = batch["input_embeddings"].to(device)
            input_texts = batch["input_texts"]
            input_lens = batch["input_lens"]
            output_embeddings = batch["output_embeddings"].to(device)
            output_texts = batch["output_texts"]
            output_lens = batch["output_lens"]
            keys = batch["keys"]
            values = batch["values"]
            
            # Get loss
            loss = model(
                input_embeddings, input_texts, input_lens,
                output_embeddings, output_texts, output_lens
            )
            
            # Generate predictions
            predictions = model.generate(input_embeddings)
            
            # Collect results
            bs = input_embeddings.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            
            for i in range(bs):
                if output_lens[i] > 0:
                    all_predictions.append(predictions[i])
                    all_references.append(values[i])
                    all_keys.append(keys[i])
                    all_values.append(values[i])
    
    # Calculate metrics
    avg_loss = total_loss / total_count if total_count > 0 else 0.0
    accuracy = string_match_all(all_predictions, all_references)
    
    # Prepare examples
    examples = []
    for i in range(min(num_examples, len(all_predictions))):
        is_correct = all_references[i].lower() in all_predictions[i].lower()
        examples.append({
            "key": all_keys[i],
            "expected_value": all_values[i],
            "prediction": all_predictions[i],
            "correct": is_correct
        })
    
    model.train()
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "examples": examples,
        "num_evaluated": total_count
    }


# ============================================================================
# Training Loop
# ============================================================================

def train_niah(config: TrainingConfig):
    """
    Main training function for NIAH task
    
    Args:
        config: Training configuration
    """
    print(f"\n{'='*80}")
    print("NIAH Training - SONAR-LLM")
    print(f"{'='*80}\n")
    
    # Set device
    if config.gpu_id is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{config.gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(config.device)
    
    print(f"Device: {device}")
    
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load data
    print(f"\nLoading data from {config.data_path}...")
    with open(config.data_path, 'r') as f:
        all_samples = json.load(f)
    
    # Split train/val
    random.shuffle(all_samples)
    val_size = int(len(all_samples) * config.val_split)
    train_samples = all_samples[val_size:]
    val_samples = all_samples[:val_size]
    
    print(f"✓ Loaded {len(all_samples)} samples")
    print(f"  Train: {len(train_samples)}")
    print(f"  Val: {len(val_samples)}")
    
    # Load SONAR pipelines
    print("\nLoading SONAR pipelines...")
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    ).eval()
    
    vec2text_model = EmbeddingToTextModelPipeline(
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    ).eval()
    
    for param in t2vec_model.parameters():
        param.requires_grad = False
    for param in vec2text_model.parameters():
        param.requires_grad = False
    
    print("✓ SONAR pipelines loaded")
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = NIAHDataset(train_samples, t2vec_model, precompute=True)
    val_dataset = NIAHDataset(val_samples, t2vec_model, precompute=True)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    print(f"✓ DataLoaders created ({len(train_loader)} train batches, {len(val_loader)} val batches)")
    
    # Load/create model
    print("\nInitializing model...")
    
    # Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    except:
        print("Using fallback tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    tokenizer.pad_token = tokenizer.eos_token
    
    # LLaMA model
    llama_config = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_layers,
        num_attention_heads=config.num_heads,
    )
    
    llama_model = LlamaForCausalLM(llama_config).to(device)
    
    # Projectors
    forward_proj = Projector(1024, config.hidden_size).to(device)
    reverse_proj = Projector(config.hidden_size, 1024).to(device)
    
    # Wrap model
    model = NIAHModel(llama_model, forward_proj, reverse_proj, vec2text_model).to(device)
    
    print(f"✓ Model initialized")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = (len(train_loader) // config.grad_accum_steps) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"✓ Optimizer and scheduler created ({total_steps} total steps)")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Initial evaluation
    print("\n" + "="*80)
    print("Initial Evaluation")
    print("="*80)
    
    init_results = evaluate_niah(model, val_loader, device, num_examples=5)
    print(f"Loss: {init_results['loss']:.4f}")
    print(f"Accuracy: {init_results['accuracy']:.2f}%")
    
    print("\nExamples:")
    for ex in init_results['examples'][:3]:
        status = "✓" if ex['correct'] else "✗"
        print(f"{status} Key={ex['key']}, Expected={ex['expected_value']}, Pred={ex['prediction'][:50]}...")
    
    # Training loop
    print("\n" + "="*80)
    print("Training")
    print("="*80 + "\n")
    
    global_step = 0
    best_accuracy = 0.0
    
    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for step, batch in enumerate(pbar):
            input_embeddings = batch["input_embeddings"].to(device)
            input_texts = batch["input_texts"]
            input_lens = batch["input_lens"]
            output_embeddings = batch["output_embeddings"].to(device)
            output_texts = batch["output_texts"]
            output_lens = batch["output_lens"]
            
            # Forward
            loss = model(
                input_embeddings, input_texts, input_lens,
                output_embeddings, output_texts, output_lens
            )
            loss = loss / config.grad_accum_steps
            
            # Backward
            loss.backward()
            epoch_loss += loss.item()
            
            # Update
            if (step + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                # Logging
                if global_step % config.logging_steps == 0:
                    pbar.set_postfix({
                        'loss': f'{epoch_loss:.4f}',
                        'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                    })
                
                # Evaluation
                if global_step % config.eval_steps == 0:
                    eval_results = evaluate_niah(model, val_loader, device, num_examples=3)
                    print(f"\n[Step {global_step}] Val Loss: {eval_results['loss']:.4f}, Accuracy: {eval_results['accuracy']:.2f}%")
                    
                    # Save best model
                    if eval_results['accuracy'] > best_accuracy:
                        best_accuracy = eval_results['accuracy']
                        save_path = os.path.join(config.output_dir, "best_model")
                        os.makedirs(save_path, exist_ok=True)
                        torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
                        print(f"  ✓ Saved best model (accuracy: {best_accuracy:.2f}%)")
                    
                    model.train()
                
                # Save checkpoint
                if global_step % config.save_steps == 0:
                    save_path = os.path.join(config.output_dir, f"checkpoint_step_{global_step}")
                    os.makedirs(save_path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
                    print(f"\n  ✓ Saved checkpoint at step {global_step}")
                
                epoch_loss = 0.0
    
    # Final evaluation
    print("\n" + "="*80)
    print("Final Evaluation")
    print("="*80)
    
    final_results = evaluate_niah(model, val_loader, device, num_examples=10)
    print(f"Loss: {final_results['loss']:.4f}")
    print(f"Accuracy: {final_results['accuracy']:.2f}%")
    
    print("\nDetailed Examples:")
    print("-"*80)
    for i, ex in enumerate(final_results['examples']):
        status = "✓ CORRECT" if ex['correct'] else "✗ INCORRECT"
        print(f"\n{status}")
        print(f"Key: {ex['key']}")
        print(f"Expected: {ex['expected_value']}")
        print(f"Prediction: {ex['prediction']}")
        print("-"*80)
    
    # Save results
    results_path = os.path.join(config.output_dir, "final_results.json")
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")
    print(f"✓ Best accuracy: {best_accuracy:.2f}%")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train SONAR-LLM on NIAH task")
    
    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--val_split", type=float, default=0.1)
    
    # Model
    parser.add_argument("--model_name", type=str, default="raxtemur/sonar-llm-1.3b")
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=10)
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="../models/niah_model")
    parser.add_argument("--gpu_id", type=int, default=0)
    
    args = parser.parse_args()
    config = TrainingConfig.from_args(args)
    
    train_niah(config)


if __name__ == "__main__":
    main()


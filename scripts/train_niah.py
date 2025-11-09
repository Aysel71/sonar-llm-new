import os
import time
import random
import datetime
import argparse
import sys
import numpy as np
import json
import re
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler, random_split

import nltk
from nltk.tokenize import sent_tokenize
# Make sure we download all needed NLTK resources
nltk.download('punkt', quiet=True)
try:
    nltk.download('punkt_tab', quiet=True)
except:
    pass  # Ignore if resource not available

import wandb

from sonar.inference_pipelines.text import (
    TextToEmbeddingModelPipeline,
    EmbeddingToTextModelPipeline
)

from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM
)

from transformers.optimization import get_cosine_schedule_with_warmup

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

# ------------------------------------------------------------------------
# 1) Parse command line arguments
# ------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps for the LR scheduler")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    
    parser.add_argument("--niah_data", type=str, required=True, 
                        help="Path to NIAH dataset file (JSON)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Fraction of data to use for validation")
    parser.add_argument("--max_train_samples", type=int, default=-1,
                        help="Use fewer samples for quick debugging. -1 for all.")
    parser.add_argument("--max_val_samples", type=int, default=-1,
                        help="Use fewer samples for quick debugging. -1 for all.")
    parser.add_argument("--output_dir", type=str, default="./niah_model")
    
    parser.add_argument("--context_length", type=int, default=512,
                        help="Maximum context length for NIAH samples")

    parser.add_argument("--use_wandb", action="store_true", help="Enable wandb logging.")
    parser.add_argument("--wandb_project", type=str, default="sonar-llm-niah", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, default="niah-training", help="Wandb run name.")

    parser.add_argument("--use_mixed_precision", action="store_true",
                        help="Use mixed precision training (fp16 autocast).")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Number of gradient accumulation steps.")

    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing (use_cache=False).")

    parser.add_argument("--start_from", type=str, default="raxtemur/sonar-llm-1.3b",
                        help="Path to a model checkpoint or huggingface model name.")
    
    parser.add_argument("--generate_niah_data", action="store_true",
                        help="Generate NIAH data if no data file is provided")
    parser.add_argument("--num_niah_samples", type=int, default=11000,
                        help="Number of NIAH samples to generate (if --generate_niah_data)")
    
    parser.add_argument("--precision", type=str, choices=["bf16", "fp16", "fp32"], default="bf16",
                       help="Precision to use for training (bf16, fp16, or fp32)")

    args = parser.parse_args()
    return args

# ------------------------------------------------------------------------
# 2) NIAH Dataset Generation
# ------------------------------------------------------------------------
def clean_text(text):
    """Clean text using similar approach to FlagEmbedding"""
    # Remove excessive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    
    # Remove special control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Convert all whitespace to standard spaces
    text = re.sub(r'\s', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def generate_distractor_text(length=1000):
    """Generate distractor text of approximate length"""
    words = [
        "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", 
        "adipiscing", "elit", "sed", "do", "eiusmod", "tempor", 
        "incididunt", "ut", "labore", "et", "dolore", "magna", 
        "aliqua", "enim", "ad", "minim", "veniam", "quis", 
        "nostrud", "exercitation", "ullamco", "laboris", "nisi",
        "ut", "aliquip", "ex", "ea", "commodo", "consequat"
    ]
    
    # Generate approximately the right amount of text
    # Assuming avg word length ~5 chars + space
    num_words = max(1, int(length / 6))
    
    # Build random sentences (5-15 words each)
    text = ""
    words_added = 0
    
    while words_added < num_words:
        sentence_length = random.randint(5, 15)
        if words_added + sentence_length > num_words:
            sentence_length = num_words - words_added
            
        sentence = " ".join(random.choice(words) for _ in range(sentence_length))
        sentence = sentence.capitalize() + ". "
        
        text += sentence
        words_added += sentence_length
    
    return text.strip()

def generate_niah_sample(context_length=512, needle_type="words"):
    """Generate a complete NIAH sample"""
    # Generate key-value for needle
    if needle_type == "numbers":
        key = f"key_{random.randint(1000, 9999)}"
        value = str(random.randint(1000000, 9999999))
    else:  # default is "words"
        adjectives = ["red", "blue", "green", "yellow", "happy", "sad", "big", "small"]
        nouns = ["dog", "cat", "house", "car", "book", "tree", "sun", "moon"]
        
        key = f"key_{random.randint(1000, 9999)}"
        value = f"{random.choice(adjectives)} {random.choice(nouns)}"
    
    # Create the needle text
    needle = f"The special magic number for {key} is: {value}."
    
    # Calculate distractor text length (approx tokens to chars)
    # Assuming 4 chars per token on average
    distractor_length = (context_length * 4) - len(needle)
    distractor_length = max(100, distractor_length)
    
    # Generate distractor text
    distractor_text = generate_distractor_text(distractor_length)
    
    # Insert needle at random position
    insertion_point = random.randint(0, len(distractor_text) - 1)
    context = distractor_text[:insertion_point] + " " + needle + " " + distractor_text[insertion_point:]
    
    # Clean the text
    cleaned_context = clean_text(context)
    
    # Create prompt and expected completion
    prompt = f"Read the following text carefully and find the value associated with {key}:\n\n{cleaned_context}\n\nWhat is the value for {key}?"
    completion = f"The value for {key} is: {value}"
    
    return {
        "input": prompt,
        "output": completion,
        "key": key,
        "value": value,
        "context_length": len(cleaned_context)
    }

def generate_niah_dataset(num_samples=11000, context_length=512):
    """Generate NIAH dataset with train/val split"""
    samples = []
    for i in tqdm(range(num_samples), desc="Generating NIAH samples"):
        sample = generate_niah_sample(context_length=context_length)
        samples.append(sample)
    
    return samples

# ------------------------------------------------------------------------
# 3) Dataset classes and data collator
# ------------------------------------------------------------------------
class NIAHDataset(Dataset):
    def __init__(self, samples, embedder, precompute=True):
        self.samples = samples
        self.embedder = embedder.eval()
        self.cache = {}
        
        # Precompute all embeddings at initialization for faster training
        if precompute:
            print(f"Precomputing embeddings for {len(samples)} samples...")
            for idx in tqdm(range(len(samples)), desc="Embedding samples"):
                self._compute_item(idx)
            print(f"Finished precomputing embeddings!")
        
    def __len__(self):
        return len(self.samples)
    
    def _compute_item(self, idx):
        """Compute embeddings for a single item"""
        if idx in self.cache:
            return self.cache[idx]
        
        sample = self.samples[idx]
        input_text = sample["input"]
        output_text = sample["output"]
        
        # Tokenize input and output into sentences
        # Using simple splitting to avoid NLTK issues
        try:
            input_sents = sent_tokenize(input_text)
        except:
            # Fallback if NLTK fails
            input_sents = [s.strip() + "." for s in input_text.split(".") if s.strip()]
            if not input_sents:
                input_sents = [input_text]
                
        try:
            output_sents = sent_tokenize(output_text)
        except:
            # Fallback if NLTK fails
            output_sents = [s.strip() + "." for s in output_text.split(".") if s.strip()]
            if not output_sents:
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
            "full_input": input_text,
            "full_output": output_text,
            "key": sample.get("key", ""),
            "value": sample.get("value", "")
        }
        
        self.cache[idx] = item
        return item
    
    def __getitem__(self, idx):
        return self._compute_item(idx)

def niah_data_collator(batch):
    # Calculate max lengths for input and output
    max_input_len = max(len(item["input_embeddings"]) for item in batch)
    max_output_len = max(len(item["output_embeddings"]) for item in batch)
    B = len(batch)
    
    # Create padded batches
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
        
        # Keys and values
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

# ------------------------------------------------------------------------
# 4) Projector and SonarLossWrapper for NIAH tasks
# ------------------------------------------------------------------------
class Projector(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
    def forward(self, x):
        return self.linear(x)

class NIAHSonarLossWrapper(nn.Module):
    def __init__(self, llama_model, forward_proj, reverse_proj, sonar_decoder):
        super().__init__()
        self.llama_model = llama_model
        self.forward_proj = forward_proj
        self.reverse_proj = reverse_proj
        self.sonar_decoder = sonar_decoder

        # Freeze the SONAR decoder
        for p in self.sonar_decoder.parameters():
            p.requires_grad = False

    def forward(self, input_embeddings, input_texts, input_lens, 
                output_embeddings, output_texts, output_lens):
        device = input_embeddings.device
        B, T_in, _ = input_embeddings.shape
        
        # Project input embeddings to Llama hidden dimension
        input_embs_projected = self.forward_proj(input_embeddings)
        
        # Get Llama hidden states
        llama_out = self.llama_model(
            inputs_embeds=input_embs_projected,
            output_hidden_states=True
        )
        last_hidden = llama_out.hidden_states[-1]
        
        # For each item in batch, collect the last hidden state of input sequence
        # This should predict the first token of output
        pred_hidden_list = []
        ref_texts_list = []
        
        for b in range(B):
            input_len = input_lens[b]
            if input_len > 0:
                pred_hidden_list.append(last_hidden[b, input_len - 1, :])
                if output_lens[b] > 0:
                    ref_texts_list.append(output_texts[b][0])
                else:
                    # Fallback if output is empty
                    ref_texts_list.append("")
        
        if len(pred_hidden_list) == 0:
            return torch.tensor(0.0, device=device)
        
        # Stack hidden states and project back to SONAR embedding space
        pred_hidden_batch = torch.stack(pred_hidden_list, dim=0)
        pred_emb_1024 = self.reverse_proj(pred_hidden_batch)
        
        # Get target text tokens using SONAR decoder
        with torch.no_grad():
            target_text_encoder = self.sonar_decoder.tokenizer.create_encoder(
                task="translation", lang="eng_Latn", mode="target", device=device
            )
        
        encoded_texts = [target_text_encoder(t) for t in ref_texts_list]
        lengths = [et.size(0) for et in encoded_texts]
        max_len = min(max(lengths) if lengths else 0, 256)
        
        if max_len == 0:
            return torch.tensor(0.0, device=device)
        
        # Prepare decoder inputs and labels
        pad_idx = self.sonar_decoder.tokenizer.vocab_info.pad_idx
        dec_ids = torch.full((len(encoded_texts), max_len), pad_idx, dtype=torch.long, device=device)
        labels = torch.full((len(encoded_texts), max_len), pad_idx, dtype=torch.long, device=device)
        
        for i, et in enumerate(encoded_texts):
            dec_ids[i, : min(len(et), max_len)] = et[:max_len]
            et_shifted = torch.cat([et[1:], torch.tensor([3]).to(device)])
            labels[i, : min(len(et_shifted), max_len)] = et_shifted[:max_len]
        
        # Run SONAR decoder on predicted embeddings
        enc_output = pred_emb_1024.unsqueeze(1)
        dec_out, dec_pad_mask = self.sonar_decoder.model.decode(
            seqs=dec_ids,
            padding_mask=None,
            encoder_output=enc_output,
            encoder_padding_mask=None
        )
        final_out = self.sonar_decoder.model.project(dec_out, dec_pad_mask)
        logits = final_out.logits
        
        # Calculate cross-entropy loss
        vocab_size = logits.size(-1)
        logits_2d = logits.view(-1, vocab_size)
        labels_1d = labels.view(-1)
        
        ce_fn = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction="mean")
        total_ce = ce_fn(logits_2d, labels_1d)
        
        return total_ce
    
    # Method for inference/validation
    def generate(self, input_embeddings, max_output_len=50):
        device = input_embeddings.device
        B, T_in, _ = input_embeddings.shape
        
        # Project to Llama hidden dimension
        input_embs_projected = self.forward_proj(input_embeddings)
        
        # Get Llama output for the last position
        with torch.no_grad():
            llama_out = self.llama_model(
                inputs_embeds=input_embs_projected,
                output_hidden_states=True
            )
            last_hidden = llama_out.hidden_states[-1]
            
            # Get the last hidden state for each batch item
            last_position_hidden = last_hidden[:, -1, :]
            
            # Project back to SONAR embedding space
            predicted_embeddings = self.reverse_proj(last_position_hidden)
            
            # Generate text using SONAR decoder
            outputs = []
            for i in range(B):
                emb = predicted_embeddings[i].unsqueeze(0)  # Shape: [1, 1024]
                
                # Use SONAR decoder to generate text
                generated_text = self.sonar_decoder.predict(
                    emb, 
                    target_lang="eng_Latn",
                    max_seq_len=max_output_len
                )[0]
                
                outputs.append(generated_text)
        
        return outputs

# ------------------------------------------------------------------------
# 5) NIAH evaluation functions
# ------------------------------------------------------------------------
def evaluate_niah_predictions(predictions, references, keys, values):
    """
    Evaluate NIAH task predictions against references
    Uses RULER-style string_match_all metric:
    - Checks if expected value appears in prediction (case-insensitive)
    - Returns accuracy of correctly identifying values for keys
    """
    correct = 0
    total = 0
    
    for pred, ref, key, value in zip(predictions, references, keys, values):
        # RULER-style evaluation: check if value is present in prediction (substring match)
        # This is more lenient than exact match
        if value.lower() in pred.lower():
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

# ------------------------------------------------------------------------
# 6) Main Training Script
# ------------------------------------------------------------------------
def main():
    args = parse_args()

    # --------------------------------------------------------------------
    # 6a) Distributed setup
    # --------------------------------------------------------------------
    local_rank = args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1

    if distributed:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f"cuda:{local_rank}")
        dist.init_process_group("nccl")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    is_main_process = (not distributed) or (dist.get_rank() == 0)
    if is_main_process:
        if distributed:
            print(f"Using {world_size} GPUs for training.")
        else:
            print("Single-process training on device:", device)
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name)
            wandb.config.update(vars(args))

    # --------------------------------------------------------------------
    # 6b) Set up precision dtype
    # --------------------------------------------------------------------
    # IMPORTANT: GradScaler only works with float16, not bfloat16!
    if args.use_mixed_precision:
        # Force fp16 when using mixed precision with GradScaler
        dtype = torch.float16
        actual_precision = "fp16"
    elif args.precision == "bf16" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
        actual_precision = "bf16"
    elif args.precision == "fp16":
        dtype = torch.float16
        actual_precision = "fp16"
    else:
        dtype = torch.float32
        actual_precision = "fp32"
        
    if is_main_process:
        print(f"Using {actual_precision} precision for training (actual dtype: {dtype})")

    # --------------------------------------------------------------------
    # 6c) Generate or load NIAH dataset
    # --------------------------------------------------------------------
    if args.generate_niah_data or not os.path.exists(args.niah_data):
        if is_main_process:
            print(f"Generating {args.num_niah_samples} NIAH samples with context length {args.context_length}")
            niah_samples = generate_niah_dataset(
                num_samples=args.num_niah_samples, 
                context_length=args.context_length
            )
            
            # Save to file
            with open(args.niah_data, 'w') as f:
                json.dump(niah_samples, f, indent=2)
            
            print(f"Saved NIAH dataset to {args.niah_data}")
        
        if distributed:
            dist.barrier()  # Wait for main process to generate data
    
    # Load NIAH dataset
    with open(args.niah_data, 'r') as f:
        niah_samples = json.load(f)
    
    if args.max_train_samples > 0 and args.max_train_samples < len(niah_samples):
        niah_samples = niah_samples[:args.max_train_samples + args.max_val_samples]
    
    # Split into train and validation
    if is_main_process:
        print(f"Loaded {len(niah_samples)} NIAH samples from {args.niah_data}")
    
    # Shuffle data
    random.shuffle(niah_samples)
    
    # Calculate split point
    val_size = min(int(len(niah_samples) * args.val_split), 
                   args.max_val_samples if args.max_val_samples > 0 else float('inf'))
    train_size = len(niah_samples) - val_size
    
    if args.max_train_samples > 0:
        train_size = min(train_size, args.max_train_samples)
    
    train_samples = niah_samples[:train_size]
    val_samples = niah_samples[train_size:train_size + val_size]
    
    if is_main_process:
        print(f"Split into {len(train_samples)} training samples and {len(val_samples)} validation samples")
        
    # --------------------------------------------------------------------
    # 6d) Model initialization
    # --------------------------------------------------------------------
    # Set up HF token from environment
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGINGFACE_TOKEN", None))
    
    # Load tokenizer from the provided model
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.start_from,
            trust_remote_code=True
        )
    except:
        # Fallback to the correct Llama-3 model with authentication
        try:
            print("Attempting to load tokenizer using Llama-3-8B-hf with authentication...")
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3-8B-hf",  # Correct model name
                token=hf_token  # Use token from environment
            )
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            print("Falling back to a public tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Configuration for model architecture (used for fallback init)
    configuration = LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=10,
        num_attention_heads=16,
        hidden_act='silu',
        max_position_embeddings=131072,
        initializer_range=0.02,
        rms_norm_eps=1e-06,
        use_cache=not args.gradient_checkpointing,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 128000,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 128001,
        pretraining_tp=1,
        tie_word_embeddings=True,
        rope_theta=500000.0,
        rope_scaling={
            "factor": 32.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        head_dim=64,
    )

    sonar_embed_dim = 1024
    llama_model = None
    forward_projector = None
    reverse_projector = None
    vec2text_model = None
    t2vec_model = None

    checkpoint_root = args.start_from
    if is_main_process:
        print(f"Attempting to load SONAR-LLM checkpoint via SONARLLMGenerator from {args.start_from}...")

    try:
        if not os.path.isdir(checkpoint_root):
            if snapshot_download is None:
                raise ImportError(
                    "huggingface_hub is required to download remote checkpoints. "
                    "Install it with `pip install huggingface_hub` or provide a local path."
                )
            snapshot_kwargs = {
                "repo_id": args.start_from,
                "repo_type": "model",
                "resume_download": True,
            }
            if hf_token:
                snapshot_kwargs["token"] = hf_token
            if is_main_process:
                print("Downloading checkpoint with huggingface_hub.snapshot_download...")
            checkpoint_root = snapshot_download(**snapshot_kwargs)
        if checkpoint_root not in sys.path:
            sys.path.insert(0, checkpoint_root)
        from sonarllm_model import SONARLLMGenerator  # type: ignore
        if is_main_process:
            print(f"Loading SONARLLMGenerator from {checkpoint_root}...")
        generator = SONARLLMGenerator.load_from_checkpoint(
            checkpoint_root,
            device=device,
        )
        llama_model = generator.llama_model.to(device).to(dtype)
        forward_projector = generator.forward_proj.to(device).to(dtype)
        reverse_projector = generator.reverse_proj.to(device).to(dtype)
        vec2text_model = generator.sonar_decoder
        t2vec_model = generator.t2vec
        if is_main_process:
            print("Successfully loaded SONARLLMGenerator checkpoint.")
    except Exception as e:
        if is_main_process:
            print(f"Failed to load SONARLLMGenerator checkpoint: {e}")
            print("Falling back to manual initialization.")
        llama_model = None
    finally:
        # Avoid keeping an extra reference to the generator wrapper
        if 'generator' in locals():
            generator = None

    if llama_model is None:
        llama_model = LlamaForCausalLM(configuration).to(device).to(dtype)
        if is_main_process:
            print("Initialized new LLaMA model with default SONAR configuration.")

    if args.gradient_checkpointing:
        llama_model.gradient_checkpointing_enable()

    if forward_projector is None:
        hidden_dim = llama_model.config.hidden_size
        forward_projector = Projector(sonar_embed_dim, hidden_dim).to(device).to(dtype)
    if reverse_projector is None:
        hidden_dim = llama_model.config.hidden_size
        reverse_projector = Projector(hidden_dim, sonar_embed_dim).to(device).to(dtype)

    # --------------------------------------------------------------------
    # 6e) Sonar pipelines (frozen)
    # --------------------------------------------------------------------
    if t2vec_model is None:
        if is_main_process:
            print("Loading SONAR text-to-embedding pipeline...")
        t2vec_model = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder",
            tokenizer="text_sonar_basic_encoder",
            device=device
        )
    else:
        if is_main_process:
            print("Reusing SONAR text-to-embedding pipeline from checkpoint.")
    t2vec_model = t2vec_model.eval()
    for param in t2vec_model.parameters():
        param.requires_grad = False

    if vec2text_model is None:
        if is_main_process:
            print("Loading SONAR embedding-to-text pipeline...")
        vec2text_model = EmbeddingToTextModelPipeline(
            decoder="text_sonar_basic_decoder",
            tokenizer="text_sonar_basic_encoder",
            device=device
        )
    else:
        if is_main_process:
            print("Reusing SONAR embedding-to-text pipeline from checkpoint.")
    vec2text_model = vec2text_model.eval()
    for param in vec2text_model.parameters():
        param.requires_grad = False

    if is_main_process:
        print("Creating projectors and model wrapper...")
    model = NIAHSonarLossWrapper(
        llama_model=llama_model,
        forward_proj=forward_projector,
        reverse_proj=reverse_projector,
        sonar_decoder=vec2text_model
    ).to(device)

    # --------------------------------------------------------------------
    # 6g) Create train and val datasets
    # --------------------------------------------------------------------
    if is_main_process:
        print("Creating training and validation datasets...")
    train_dataset = NIAHDataset(train_samples, t2vec_model)
    val_dataset = NIAHDataset(val_samples, t2vec_model)

    if is_main_process:
        print(f"Train size = {len(train_dataset)}, Val size = {len(val_dataset)}")
        if args.use_wandb:
            wandb.log({"train/total_samples": len(train_dataset)})

    # --------------------------------------------------------------------
    # 6h) DataLoaders
    # --------------------------------------------------------------------
    if is_main_process:
        print("Creating DataLoaders...")
    
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_dloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=0,
        collate_fn=niah_data_collator,
        drop_last=False
    )
    
    if is_main_process:
        print(f"Training DataLoader created with {len(train_dloader)} batches")
    
    val_dloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        collate_fn=niah_data_collator,
        drop_last=False
    )
    
    if is_main_process:
        print(f"Validation DataLoader created with {len(val_dloader)} batches")

    # --------------------------------------------------------------------
    # 6i) Wrap in DDP if needed
    # --------------------------------------------------------------------
    if is_main_process:
        print("Setting up distributed training (if applicable)...")
    
    if distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )

    # --------------------------------------------------------------------
    # 6j) Optimizer & Scheduler
    # --------------------------------------------------------------------
    if is_main_process:
        print("Creating optimizer and scheduler...")
    
    no_decay = ["bias", "LayerNorm.weight"]
    named_params = list(model.named_parameters())  # DDP -> (module.xxx, param)
    optimizer_params = []
    for name, param in named_params:
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay):
            optimizer_params.append({"params": [param], "weight_decay": 0.0})
        else:
            optimizer_params.append({"params": [param], "weight_decay": args.weight_decay})
    optimizer = torch.optim.AdamW(optimizer_params, lr=args.lr)
    
    if is_main_process:
        print(f"Optimizer created with {len(optimizer_params)} parameter groups")

    steps_per_epoch = len(train_dloader)
    total_steps = (steps_per_epoch // args.grad_accum_steps) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps
    )
    
    if is_main_process:
        print(f"Scheduler created: {total_steps} total steps, {args.warmup_steps} warmup steps")

    # --------------------------------------------------------------------
    # 6k) Mixed Precision + GradScaler
    # --------------------------------------------------------------------
    # Fixed: Use torch.amp.GradScaler with device_type
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_mixed_precision)
    
    if is_main_process:
        print("GradScaler created")

    # --------------------------------------------------------------------
    # 6l) Training Loop
    # --------------------------------------------------------------------
    model.train()
    full_loss = 0.0
    global_step = 0

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    if is_main_process:
        print(f"Output directory created: {args.output_dir}")

    # Initial evaluation
    if is_main_process:
        print("Running initial validation...")
        val_loss, val_accuracy, examples = evaluate_niah(model, val_dloader, device, distributed, dtype, show_examples=4)
        print(f"Initial validation: loss={val_loss:.4f}, accuracy={val_accuracy:.4f}")
        print("\n" + "="*80)
        print("EVALUATION EXAMPLES:")
        print("="*80)
        for ex in examples:
            status = "✓ CORRECT" if ex["correct"] else "✗ INCORRECT"
            print(f"\n{status}")
            print(f"Key: {ex['key']}")
            print(f"Expected value: {ex['expected_value']}")
            print(f"Reference: {ex['reference']}")
            print(f"Prediction: {ex['prediction']}")
            print("-"*80)
        print("="*80 + "\n")
        
        if args.use_wandb:
            wandb.log({"eval/loss": val_loss, "eval/accuracy": val_accuracy, "eval/step": 0})
    else:
        val_loss, val_accuracy = evaluate_niah(model, val_dloader, device, distributed, dtype, show_examples=0)

    for epoch in range(args.epochs):
        if distributed:
            train_dloader.sampler.set_epoch(epoch)

        for step, batch in enumerate(tqdm(train_dloader, desc=f"Epoch {epoch+1}")):
            # Move batch to device
            input_embeddings = batch["input_embeddings"].to(device)
            input_texts = batch["input_texts"]
            input_lens = batch["input_lens"]
            
            output_embeddings = batch["output_embeddings"].to(device)
            output_texts = batch["output_texts"]
            output_lens = batch["output_lens"]

            # Forward pass with mixed precision
            if args.use_mixed_precision:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    loss = model(
                        input_embeddings, input_texts, input_lens,
                        output_embeddings, output_texts, output_lens
                    )
                loss = loss / args.grad_accum_steps
                full_loss += loss.item()
                
                # Backward with gradient scaling (outside autocast!)
                scaler.scale(loss).backward()
            else:
                # No mixed precision
                loss = model(
                    input_embeddings, input_texts, input_lens,
                    output_embeddings, output_texts, output_lens
                )
                loss = loss / args.grad_accum_steps
                full_loss += loss.item()
                loss.backward()

            # Optimizer step
            if (step + 1) % args.grad_accum_steps == 0:
                if args.use_mixed_precision:
                    # Unscale gradients and clip
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Optimizer step with scaler
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular gradient clipping and step
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if (global_step % args.logging_steps == 0) and is_main_process:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"[Epoch {epoch+1}, Step {global_step}] loss = {full_loss:.4f}, lr = {current_lr:.6f}")
                    if args.use_wandb:
                        samples_seen = global_step * args.batch_size
                        if distributed:
                            samples_seen *= world_size
                        wandb.log({
                            "train/loss": full_loss,
                            "train/step": global_step,
                            "train/samples_seen": samples_seen,
                            "train/lr": current_lr,
                        })
                full_loss = 0.0

                if (global_step % args.eval_steps == 0):
                    if is_main_process:
                        val_loss, val_accuracy, examples = evaluate_niah(model, val_dloader, device, distributed, dtype, show_examples=4)
                        print(f"[Eval] global_step={global_step}, val_loss={val_loss:.4f}, val_accuracy={val_accuracy:.4f}")
                        print("\n" + "-"*80)
                        print("EVALUATION EXAMPLES:")
                        print("-"*80)
                        for ex in examples:
                            status = "✓" if ex["correct"] else "✗"
                            print(f"{status} Key={ex['key']}, Expected={ex['expected_value']}")
                            print(f"  Pred: {ex['prediction'][:100]}...")  # Show first 100 chars
                        print("-"*80 + "\n")
                        
                        if args.use_wandb:
                            wandb.log({
                                "eval/loss": val_loss,
                                "eval/accuracy": val_accuracy, 
                                "eval/step": global_step
                            })
                    else:
                        val_loss, val_accuracy = evaluate_niah(model, val_dloader, device, distributed, dtype, show_examples=0)

                if (global_step % args.save_steps == 0) and is_main_process:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        tokenizer=tokenizer,
                        epoch=epoch,
                        global_step=global_step,
                        output_dir=args.output_dir
                    )

        # Save a checkpoint at the end of each epoch
        if is_main_process:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                tokenizer=tokenizer,
                epoch=epoch + 1,
                global_step=global_step,
                output_dir=args.output_dir
            )

    if distributed:
        dist.barrier()
        
    # Final evaluation on the full validation set
    if is_main_process:
        val_loss, val_accuracy, examples = evaluate_niah(model, val_dloader, device, distributed, dtype, show_examples=10)
        print("\n" + "="*80)
        print("FINAL EVALUATION RESULTS")
        print("="*80)
        print(f"Loss: {val_loss:.4f}")
        print(f"Accuracy: {val_accuracy:.2%}")
        print("\n" + "="*80)
        print("DETAILED EXAMPLES:")
        print("="*80)
        for ex in examples:
            status = "✓ CORRECT" if ex["correct"] else "✗ INCORRECT"
            print(f"\n{status}")
            print(f"Key: {ex['key']}")
            print(f"Expected value: {ex['expected_value']}")
            print(f"Reference: {ex['reference']}")
            print(f"Prediction: {ex['prediction']}")
            print("-"*80)
        print("="*80 + "\n")
        
        # Save final metrics
        results = {
            "final_loss": val_loss,
            "final_accuracy": val_accuracy,
            "num_train_samples": len(train_dataset),
            "num_val_samples": len(val_dataset),
            "epochs": args.epochs,
            "global_steps": global_step,
            "examples": examples
        }
        
        with open(os.path.join(args.output_dir, "final_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Final results saved to {args.output_dir}/final_results.json")
    else:
        val_loss, val_accuracy = evaluate_niah(model, val_dloader, device, distributed, dtype, show_examples=0)

# ------------------------------------------------------------------------
# 7) NIAH Evaluation function
# ------------------------------------------------------------------------
def evaluate_niah(model, val_dloader, device, distributed, dtype=None, show_examples=0):
    """
    Evaluate NIAH model on validation set
    
    Args:
        show_examples: Number of example cases to return (0 = don't return examples)
    
    Returns:
        loss, accuracy, examples (if show_examples > 0)
    """
    model.eval()

    total_loss = 0.0
    total_count = 0
    
    all_predictions = []
    all_references = []
    all_keys = []
    all_values = []
    all_correct = []
    
    with torch.no_grad():
        for batch in val_dloader:
            # Move batch to device
            input_embeddings = batch["input_embeddings"].to(device)
            input_texts = batch["input_texts"]
            input_lens = batch["input_lens"]
            
            output_embeddings = batch["output_embeddings"].to(device)
            output_texts = batch["output_texts"]
            output_lens = batch["output_lens"]
            
            keys = batch["keys"]
            values = batch["values"]

            # Evaluation forward pass
            if dtype == torch.float16:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Get loss
                    loss = model(
                        input_embeddings, input_texts, input_lens,
                        output_embeddings, output_texts, output_lens
                    )
                    
                    # Generate predictions
                    raw_model = model.module if hasattr(model, "module") else model
                    predictions = raw_model.generate(input_embeddings)
            else:
                # Get loss
                loss = model(
                    input_embeddings, input_texts, input_lens,
                    output_embeddings, output_texts, output_lens
                )
                
                # Generate predictions
                raw_model = model.module if hasattr(model, "module") else model
                predictions = raw_model.generate(input_embeddings)

            bs = input_embeddings.size(0)
            total_loss += loss.item() * bs
            total_count += bs
            
            # Collect predictions and references
            for i in range(bs):
                if output_lens[i] > 0:
                    pred = predictions[i]
                    ref = " ".join(output_texts[i][:output_lens[i]])
                    key = keys[i]
                    value = values[i]
                    
                    # Check if correct (RULER-style substring match)
                    is_correct = value.lower() in pred.lower()
                    
                    all_predictions.append(pred)
                    all_references.append(ref)
                    all_keys.append(key)
                    all_values.append(value)
                    all_correct.append(is_correct)

    if distributed:
        # Gather loss and count
        result = torch.tensor([total_loss, total_count], device=device, dtype=torch.float32)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        total_loss, total_count = result[0].item(), result[1].item()
        
        # Gather predictions and references
        # This is simplified and would need more work for a complete implementation
        # For now, we just use the predictions from rank 0

    # Calculate metrics
    loss = (total_loss / total_count) if total_count > 0 else 0.0
    
    # Calculate accuracy
    accuracy, correct, total = evaluate_niah_predictions(
        all_predictions, all_references, all_keys, all_values
    )

    model.train()
    
    # Prepare examples if requested
    if show_examples > 0:
        examples = []
        # Show some correct and some incorrect examples
        correct_examples = [(i, p, r, k, v) for i, (p, r, k, v, c) in enumerate(zip(
            all_predictions, all_references, all_keys, all_values, all_correct
        )) if c]
        incorrect_examples = [(i, p, r, k, v) for i, (p, r, k, v, c) in enumerate(zip(
            all_predictions, all_references, all_keys, all_values, all_correct
        )) if not c]
        
        # If we have both correct and incorrect, show equal amounts
        if len(correct_examples) > 0 and len(incorrect_examples) > 0:
            n_each = min(show_examples // 2, len(correct_examples), len(incorrect_examples))
            
            for idx, pred, ref, key, value in correct_examples[:n_each]:
                examples.append({
                    "index": idx,
                    "key": key,
                    "expected_value": value,
                    "reference": ref,
                    "prediction": pred,
                    "correct": True
                })
            
            for idx, pred, ref, key, value in incorrect_examples[:n_each]:
                examples.append({
                    "index": idx,
                    "key": key,
                    "expected_value": value,
                    "reference": ref,
                    "prediction": pred,
                    "correct": False
                })
        else:
            # If all correct or all incorrect, just show any examples
            all_examples = [(i, p, r, k, v, c) for i, (p, r, k, v, c) in enumerate(zip(
                all_predictions, all_references, all_keys, all_values, all_correct
            ))]
            
            for idx, pred, ref, key, value, is_correct in all_examples[:show_examples]:
                examples.append({
                    "index": idx,
                    "key": key,
                    "expected_value": value,
                    "reference": ref,
                    "prediction": pred,
                    "correct": is_correct
                })
        
        return loss, accuracy, examples
    
    return loss, accuracy

# ------------------------------------------------------------------------
# 8) Checkpoint Save
# ------------------------------------------------------------------------
def save_checkpoint(model, optimizer, scheduler, tokenizer, epoch, global_step, output_dir):
    """Save model, optimizer, and scheduler states"""
    os.makedirs(output_dir, exist_ok=True)
    raw_model = model.module if hasattr(model, "module") else model

    # Save the full model (Llama + projectors)
    checkpoint_path = os.path.join(output_dir, f"checkpoint_step_{global_step}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save SONAR-LLM components separately
    # 1. Save Llama model and tokenizer
    llama_path = os.path.join(checkpoint_path, "llama")
    os.makedirs(llama_path, exist_ok=True)
    raw_model.llama_model.save_pretrained(llama_path)
    tokenizer.save_pretrained(llama_path)
    
    # 2. Save projectors
    projector_path = os.path.join(checkpoint_path, "projectors.pt")
    torch.save({
        "forward_proj": raw_model.forward_proj.state_dict(),
        "reverse_proj": raw_model.reverse_proj.state_dict()
    }, projector_path)
    
    # 3. Save optimizer and scheduler
    training_path = os.path.join(checkpoint_path, "training_state.pt")
    torch.save({
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_step": global_step
    }, training_path)
    
    print(f"Saved checkpoint to {checkpoint_path}")

# ------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------
if __name__ == "__main__":
    main()

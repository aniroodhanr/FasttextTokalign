"""Run vanilla alignment evaluation"""

import json
import numpy as np
from transformers import AutoTokenizer

from src.embeddings import build_fasttext_embeddings
from src.alignment import compute_vanilla_alignment
from src.evaluation import evaluate_bleu_bertscore, print_results


def main():
    # Configuration
    SOURCE_MODEL = "EleutherAI/pythia-1b"
    TARGET_MODEL = "Qwen/Qwen2-1.5B"
    
    print("="*70)
    print("VANILLA ALIGNMENT EVALUATION")
    print("="*70)
    print(f"\nSource: {SOURCE_MODEL}")
    print(f"Target: {TARGET_MODEL}\n")
    
    # Load tokenizers
    print("→ Loading tokenizers...")
    src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    
    # Build embeddings
    print("\n→ Building embeddings...")
    src_emb = build_fasttext_embeddings(src_tok, "pythia_vanilla", force_train=False)
    tgt_emb = build_fasttext_embeddings(tgt_tok, "qwen_vanilla", force_train=False)
    
    # Compute alignment
    print("\n→ Computing alignment...")
    token_map, align_matrix = compute_vanilla_alignment(src_emb, tgt_emb)
    
    # Save alignment
    print("\n→ Saving alignment...")
    np.save("results/vanilla_alignment_matrix.npy", align_matrix)
    np.save("results/vanilla_token_map.npy", token_map)
    
    # Evaluate
    print("\n→ Evaluating...")
    results = evaluate_bleu_bertscore(token_map, src_tok, tgt_tok, 
                                     test_size=500, compute_bert=True)
    
    # Add metadata
    results['method'] = 'vanilla'
    results['source_model'] = SOURCE_MODEL
    results['target_model'] = TARGET_MODEL
    
    # Print results
    print_results(results, "Vanilla Alignment")
    
    # Save results
    with open("results/vanilla_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✓ Saved results to results/vanilla_results.json")


if __name__ == "__main__":
    main()
"""Run relative representation alignment evaluation"""

import json
import numpy as np
from transformers import AutoTokenizer

from src.embeddings import build_fasttext_embeddings
from src.alignment import compute_relative_alignment
from src.evaluation import evaluate_bleu_bertscore, print_results


def main():
    # Configuration
    SOURCE_MODEL = "EleutherAI/pythia-1b"
    TARGET_MODEL = "Qwen/Qwen2-1.5B"
    N_ANCHORS = 300
    
    print("="*70)
    print("RELATIVE REPRESENTATION ALIGNMENT EVALUATION")
    print("="*70)
    print(f"\nSource: {SOURCE_MODEL}")
    print(f"Target: {TARGET_MODEL}")
    print(f"Anchors: {N_ANCHORS}\n")
    
    # Load tokenizers
    print("→ Loading tokenizers...")
    src_tok = AutoTokenizer.from_pretrained(SOURCE_MODEL)
    tgt_tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
    
    # Build embeddings
    print("\n→ Building embeddings...")
    src_emb = build_fasttext_embeddings(src_tok, "pythia_relative", force_train=False)
    tgt_emb = build_fasttext_embeddings(tgt_tok, "qwen_relative", force_train=False)
    
    # Compute alignment
    print("\n→ Computing alignment...")
    token_map, align_matrix, anchors = compute_relative_alignment(
        src_emb, tgt_emb, src_tok, tgt_tok, n_anchors=N_ANCHORS
    )
    
    # Save alignment
    print("\n→ Saving alignment...")
    np.save("results/relative_alignment_matrix.npy", align_matrix)
    np.save("results/relative_token_map.npy", token_map)
    
    with open("results/relative_anchors.json", "w") as f:
        json.dump({'anchors': anchors, 'n_anchors': len(anchors)}, f, indent=2)
    
    # Evaluate
    print("\n→ Evaluating...")
    results = evaluate_bleu_bertscore(token_map, src_tok, tgt_tok,
                                     test_size=500, compute_bert=True)
    
    # Add metadata
    results['method'] = 'relative'
    results['source_model'] = SOURCE_MODEL
    results['target_model'] = TARGET_MODEL
    results['n_anchors'] = len(anchors)
    
    # Print results
    print_results(results, "Relative Alignment")
    
    # Save results
    with open("results/relative_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✓ Saved results to results/relative_results.json")


if __name__ == "__main__":
    main()
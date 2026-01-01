

import numpy as np


def compute_vanilla_alignment(src_embeddings, tgt_embeddings):
  
    print("→ Computing VANILLA alignment...")
    
    # Normalize embeddings
    src_norm = src_embeddings / (np.linalg.norm(src_embeddings, axis=1, keepdims=True) + 1e-10)
    tgt_norm = tgt_embeddings / (np.linalg.norm(tgt_embeddings, axis=1, keepdims=True) + 1e-10)
    
    # Compute cosine similarity matrix
    align_matrix = np.dot(tgt_norm, src_norm.T)
    
    # Create token mapping
    token_map = np.argmax(align_matrix, axis=1)
    
    print(f"✓ Alignment matrix shape: {align_matrix.shape}")
    print(f"✓ Mean similarity: {align_matrix.max(axis=1).mean():.4f}")
    
    return token_map, align_matrix


def find_anchor_tokens(src_tokenizer, tgt_tokenizer, n_anchors=300):
    
    print(f"→ Finding {n_anchors} anchor tokens...")
    
    # Find common tokens
    src_vocab = src_tokenizer.get_vocab()
    tgt_vocab = tgt_tokenizer.get_vocab()
    common = list(set(src_vocab.keys()) & set(tgt_vocab.keys()))
    
    # Filter: alphabetic and length > 2
    valid_anchors = sorted([t for t in common if len(t) > 2 and t.isalpha()])
    
    # Select first n_anchors
    anchors = valid_anchors[:n_anchors]
    
    # Get token IDs
    src_anchor_ids = [src_vocab[t] for t in anchors]
    tgt_anchor_ids = [tgt_vocab[t] for t in anchors]
    
    print(f" Selected {len(anchors)} anchors")
    print(f"  Sample anchors: {anchors[:10]}")
    
    return anchors, src_anchor_ids, tgt_anchor_ids


def compute_relative_alignment(src_embeddings, tgt_embeddings, 
                               src_tokenizer, tgt_tokenizer, n_anchors=300):
    
    print("→ Computing RELATIVE alignment...")
    
    # Find anchor tokens
    anchors, src_ids, tgt_ids = find_anchor_tokens(
        src_tokenizer, tgt_tokenizer, n_anchors
    )
    
    # Compute relative representations
    print("→ Computing relative representations...")
    src_relative = np.dot(src_embeddings, src_embeddings[src_ids].T)
    tgt_relative = np.dot(tgt_embeddings, tgt_embeddings[tgt_ids].T)
    
    # Normalize
    src_norm = src_relative / (np.linalg.norm(src_relative, axis=1, keepdims=True) + 1e-10)
    tgt_norm = tgt_relative / (np.linalg.norm(tgt_relative, axis=1, keepdims=True) + 1e-10)
    
    # Compute alignment
    align_matrix = np.dot(tgt_norm, src_norm.T)
    token_map = np.argmax(align_matrix, axis=1)
    
    print(f"Alignment matrix shape: {align_matrix.shape}")
    print(f"Mean similarity: {align_matrix.max(axis=1).mean():.4f}")
    
    return token_map, align_matrix, anchors
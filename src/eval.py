

import numpy as np
import torch
from tqdm import tqdm
from sacrebleu import corpus_bleu
from datasets import load_dataset
from bert_score import BERTScorer


def evaluate_bleu_bertscore(token_map, src_tokenizer, tgt_tokenizer, 
                            test_size=500, compute_bert=True):
    """
    Evaluate alignment quality using BLEU and BERTScore.
    
    Args:
        token_map: Token mapping array (tgt_vocab_size,)
        src_tokenizer: Source tokenizer
        tgt_tokenizer: Target tokenizer
        test_size: Number of test samples
        compute_bert: Whether to compute BERTScore (slow)
        
    Returns:
        dict: Results containing BLEU and BERTScore metrics
    """
    print(f"→ Evaluating on {test_size} test samples...")
    
    # Load test data
    test_data = load_dataset("wikitext", "wikitext-103-raw-v1", 
                            split=f"test[:{test_size}]")
    
    hyps, refs = [], []
    decoded_hyps, decoded_refs = [], []
    
    # Process test samples
    for entry in tqdm(test_data, desc="  Processing"):
        text = entry["text"].strip()
        if not text:
            continue
        
        # Tokenize with both tokenizers
        s_ids = src_tokenizer.encode(text, add_special_tokens=False)
        t_ids = tgt_tokenizer.encode(text, add_special_tokens=False)
        
        # Apply alignment
        pred_ids = [token_map[sid] for sid in s_ids if sid < len(token_map)]
        
        # For BLEU
        hyps.append(" ".join(map(str, pred_ids)))
        refs.append([" ".join(map(str, t_ids))])
        
        # For BERTScore
        if compute_bert:
            try:
                pred_text = tgt_tokenizer.decode(pred_ids, skip_special_tokens=True)
                decoded_hyps.append(pred_text if pred_text.strip() else "empty")
                decoded_refs.append(text if text.strip() else "empty")
            except:
                decoded_hyps.append("empty")
                decoded_refs.append("empty")
    
    # Compute BLEU
    print("→ Computing BLEU...")
    bleu = corpus_bleu(hyps, refs)
    
    results = {
        'bleu': {
            'bleu1': float(bleu.precisions[0]),
            'bleu2': float(bleu.precisions[1]),
            'bleu3': float(bleu.precisions[2]),
            'bleu4': float(bleu.precisions[3]),
            'overall': float(bleu.score)
        },
        'test_samples': len(hyps)
    }
    
    # Compute BERTScore if requested
    if compute_bert:
        print("→ Computing BERTScore...")
        scorer = BERTScorer(
            model_type="bert-base-uncased",
            lang="en",
            batch_size=32,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        batch_size = 100
        all_P, all_R, all_F1 = [], [], []
        
        for i in tqdm(range(0, len(decoded_hyps), batch_size), desc="  BERTScore"):
            batch_hyps = decoded_hyps[i:i+batch_size]
            batch_refs = decoded_refs[i:i+batch_size]
            
            P, R, F1 = scorer.score(batch_hyps, batch_refs)
            all_P.extend(P.cpu().numpy())
            all_R.extend(R.cpu().numpy())
            all_F1.extend(F1.cpu().numpy())
        
        results['bertscore'] = {
            'precision': float(np.mean(all_P)),
            'recall': float(np.mean(all_R)),
            'f1': float(np.mean(all_F1))
        }
    
    return results


def print_results(results, method_name="Alignment"):
    """Pretty print evaluation results."""
    print("\n" + "="*65)
    print(f"{method_name.upper()} RESULTS")
    print("="*65)
    
    print(f"\nBLEU Scores:")
    print(f"  BLEU-1:          {results['bleu']['bleu1']:.2f}")
    print(f"  BLEU-2:          {results['bleu']['bleu2']:.2f}")
    print(f"  BLEU-3:          {results['bleu']['bleu3']:.2f}")
    print(f"  BLEU-4:          {results['bleu']['bleu4']:.2f}")
    print(f"  Overall BLEU:    {results['bleu']['overall']:.2f}")
    
    if 'bertscore' in results:
        print(f"\nBERTScore:")
        print(f"  Precision:       {results['bertscore']['precision']:.4f}")
        print(f"  Recall:          {results['bertscore']['recall']:.4f}")
        print(f"  F1:              {results['bertscore']['f1']:.4f}")
    
    print("="*65 + "\n")
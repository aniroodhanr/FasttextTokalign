"""Compare vanilla and relative alignment results"""

import json


def main():
    # Load results
    with open("results/vanilla_results.json") as f:
        vanilla = json.load(f)
    
    with open("results/relative_results.json") as f:
        relative = json.load(f)
    
    # Print comparison
    print("\n" + "="*70)
    print("VANILLA vs RELATIVE COMPARISON")
    print("="*70)
    print(f"\n{'Metric':<20} {'Vanilla':<15} {'Relative':<15} {'Improvement'}")
    print("-"*70)
    
    # BLEU scores
    for metric in ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'overall']:
        v = vanilla['bleu'][metric]
        r = relative['bleu'][metric]
        imp = ((r - v) / v * 100) if v > 0 else 0
        label = metric.upper() if metric != 'overall' else 'Overall BLEU'
        print(f"{label:<20} {v:<15.2f} {r:<15.2f} {imp:>10.2f}%")
    
    # BERTScore
    if 'bertscore' in vanilla and 'bertscore' in relative:
        print()
        for metric in ['precision', 'recall', 'f1']:
            v = vanilla['bertscore'][metric]
            r = relative['bertscore'][metric]
            imp = ((r - v) / v * 100) if v > 0 else 0
            label = f"BERTScore {metric.capitalize()}"
            print(f"{label:<20} {v:<15.4f} {r:<15.4f} {imp:>10.2f}%")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    main()

import os
import numpy as np
import fasttext
from datasets import load_dataset


def build_fasttext_embeddings(tokenizer, name, corpus_size=30000, dim=100, force_train=False):
    f True, retrain even if cached model exists
        
    
    model_path = f"{name}_model.bin"
    
    # Load cached model if exists
    if os.path.exists(model_path) and not force_train:
        print(f"→ Loading cached {name} embeddings...", flush=True)
        ft_model = fasttext.load_model(model_path)
    else:
        # Train new FastText model
        print(f"→ Building {name} embeddings...", flush=True)
        
        # Load corpus
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", 
                             split=f"train[:{corpus_size}]")
        
        # Create corpus file
        corpus_file = f"{name}_corpus.txt"
        with open(corpus_file, "w", encoding="utf-8") as f:
            for line in dataset["text"]:
                if len(line.strip()) > 10:
                    f.write(line + "\n")
        
        # Train FastText
        ft_model = fasttext.train_unsupervised(
            corpus_file,
            model='skipgram',
            dim=dim
        )
        ft_model.save_model(model_path)
        
        # Cleanup corpus file
        os.remove(corpus_file)
    
    # Extract embeddings for vocabulary
    vocab = tokenizer.get_vocab()
    vectors = np.zeros((len(vocab), dim))
    
    for word, idx in vocab.items():
        vectors[idx] = ft_model.get_word_vector(word)
    
    print(f"✓ Embeddings shape: {vectors.shape}")
    return vectors
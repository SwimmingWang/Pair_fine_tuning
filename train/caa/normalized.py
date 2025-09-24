from behaviors import ALL_BEHAVIORS, get_vector_path
import torch as t
import os
import argparse
from typing import List

def normalize_vectors(model_name_path: str, n_layers: int, behaviors: List[str]):
    # make normalized_vectors directory
    # normalized_vectors_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalized_vectors")
    # if not os.path.exists(normalized_vectors_dir):
    #     os.makedirs(normalized_vectors_dir)
    for layer in range(n_layers):
        print(layer)
        norms = {}
        vecs = {}
        new_paths = {}
        for behavior in behaviors:
            vec_path = get_vector_path(behavior, layer, model_name_path)
            vec = t.load(vec_path)
            norm = vec.norm().item()
            vecs[behavior] = vec
            norms[behavior] = norm
            new_path = vec_path.replace("vectors", "normalized_vectors")
            new_paths[behavior] = new_path
        # print(norms)
        mean_norm = t.tensor(list(norms.values())).mean().item()
        # normalize all vectors to have the same norm
        for behavior in behaviors:
            vecs[behavior] = vecs[behavior] * mean_norm / norms[behavior]
        # save the normalized vectors
        for behavior in behaviors:
            if not os.path.exists(os.path.dirname(new_paths[behavior])):
                os.makedirs(os.path.dirname(new_paths[behavior]))
            t.save(vecs[behavior], new_paths[behavior])
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_path", type=str, required=True)
    parser.add_argument("--n_layers", type=int, required=True)
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    args = parser.parse_args()
    normalize_vectors(args.model_name_path, args.n_layers, args.behaviors)
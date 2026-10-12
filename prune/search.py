"""MAC-constrained mask derivation and SAP spectral library retrieval."""
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import torch

from prune.efficiency.mac import compute_mask_mac, mac_per_head, mac_per_neuron


def model_config(args):
    return {'N_dec': args.N_dec, 'd_model': args.d_model, 'h': args.h,
            'dropout': 0, 'dropout_attn': 0}


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def spectral_signature(parity):
    """Top two algebraic eigenvalues of [[0, H.T], [H, 0]].

    The positive eigenvalues of this bipartite adjacency matrix are H's
    singular values, so constructing the larger adjacency matrix is unnecessary.
    """
    matrix = np.asarray(parity, dtype=np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    return torch.tensor(np.pad(singular_values[:2], (0, max(0, 2 - len(singular_values)))),
                        dtype=torch.float64)


def validate_masks(head_mask, neuron_mask, config):
    for mask, shape in ((head_mask, (config['N_dec'], config['h'])),
                        (neuron_mask, (config['N_dec'], config['d_model'] * 4))):
        if not isinstance(mask, torch.Tensor) or tuple(mask.shape) != shape:
            raise ValueError('Mask and model dimensions do not match')
        if not torch.all((mask == 0) | (mask == 1)):
            raise ValueError('Masks must be binary')
    if not torch.all(head_mask.sum(dim=1) >= 1):
        raise ValueError('Every layer must retain at least one attention head')


def check_ratio(library, pruning_ratio):
    # Strict equality is intentional: never retrieve a mask from a different budget.
    if not 0 <= pruning_ratio < 1 or library['model_prun_ratio'] != pruning_ratio:
        raise ValueError('Pruning ratio does not match the library')


def load_library(path, pruning_ratio, config):
    library = torch.load(path, map_location='cpu', weights_only=True)
    check_ratio(library, pruning_ratio)
    if library['model_config'] != config:
        raise ValueError('Library and model configurations do not match')
    if not isinstance(library['checkpoint_sha256'], str) or not library['entries']:
        raise ValueError('Invalid or empty library')
    for entry in library['entries']:
        signature = entry['signature']
        if not isinstance(signature, torch.Tensor) or signature.shape != (2,) or not signature.isfinite().all():
            raise ValueError('Invalid spectral signature')
        validate_masks(entry['head_mask'], entry['neuron_mask'], config)
    return library


def retrieve_mask(library, parity, pruning_ratio):
    check_ratio(library, pruning_ratio)
    signature = spectral_signature(parity)
    entry = min(library['entries'], key=lambda e: torch.linalg.vector_norm(e['signature'] - signature).item())
    distance = torch.linalg.vector_norm(entry['signature'] - signature).item()
    return entry if math.exp(-0.1 * distance) >= 0.5 else None


def save_library(library, path):
    """Keep the previous complete library intact if saving is interrupted."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.', suffix='.tmp', delete=False) as handle:
            temporary = Path(handle.name)
            torch.save(library, handle)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@torch.no_grad()
def search_mac_ecc_modify(cfg, head_grads, neuron_grads, seq_len, mac_budget_ratio):
    """Original Fisher/MAC greedy search, retaining one head in each layer."""
    if not 0 < mac_budget_ratio <= 1:
        raise ValueError('Invalid MAC budget')
    if (not head_grads.isfinite().all() or not neuron_grads.isfinite().all()
            or (head_grads < 0).any() or (neuron_grads < 0).any()):
        raise ValueError('Invalid Fisher scores')
    device = head_grads.device
    heads, neurons = head_grads.cpu(), neuron_grads.cpu()
    layers, num_heads = heads.shape
    head_mask, neuron_mask = torch.zeros_like(heads), torch.zeros_like(neurons)
    head_mask[torch.arange(layers), heads.argmax(dim=1)] = 1
    used_mac, full_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, cfg.hidden_size)
    used_mac, limit = float(used_mac), float(full_mac) * mac_budget_ratio
    if used_mac > limit:
        raise ValueError('MAC budget cannot retain one head per layer')
    head_cost = mac_per_head(seq_len, cfg.hidden_size, cfg.hidden_size // num_heads)
    neuron_cost = mac_per_neuron(seq_len, cfg.hidden_size)
    head_order = [i for i in heads.flatten().argsort(descending=True).tolist() if not head_mask.flatten()[i]]
    neuron_order = neurons.flatten().argsort(descending=True).tolist()
    head_scores, neuron_scores = heads.flatten().tolist(), neurons.flatten().tolist()
    hi = ni = 0
    while hi < len(head_order) or ni < len(neuron_order):
        head_gain = (head_scores[head_order[hi]] / head_cost
                     if hi < len(head_order) and used_mac + head_cost <= limit else -1)
        neuron_gain = (neuron_scores[neuron_order[ni]] / neuron_cost
                       if ni < len(neuron_order) and used_mac + neuron_cost <= limit else -1)
        if head_gain < 0 and neuron_gain < 0:
            break
        if head_gain >= neuron_gain:
            head_mask.view(-1)[head_order[hi]] = 1
            used_mac += head_cost
            hi += 1
        else:
            neuron_mask.view(-1)[neuron_order[ni]] = 1
            used_mac += neuron_cost
            ni += 1
    return head_mask.to(device), neuron_mask.to(device)

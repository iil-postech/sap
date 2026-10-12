"""
Official implementation of "Spectral-Aligned Pruning for Universal Error-Correcting Code Transformers" (SAP)
Information and Intelligence Lab (IIL)
Department of Electrical Engineering
Pohang University of Science and Technology (POSTECH), South Korea.
@author: Cho Sanghyeon, chosh320@postech.ac.kr
"""
import argparse
import copy
import logging
import os
import pickle
from pathlib import Path
import random
import re
import time
import warnings
from types import SimpleNamespace

import loralib as lora
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from Codes import BER, FER, EbN0_to_std, bin_to_sign, get_generator, read_parity_matrix, sign_to_bin
from Model_DC import DC_ECC_Transformer, prune_linear_layer
from prune.fisher import collect_mask_grads_ecc_revised
from prune.search import (
    checkpoint_digest, load_library, model_config, retrieve_mask, save_library,
    search_mac_ecc_modify, spectral_signature, validate_masks,
)

ROOT = Path(__file__).resolve().parent
PRETRAINED_PATH = ROOT / 'pretrained_fecct.pt'
PROVIDED_PRETRAINED_PATH = ROOT / 'checkpoints' / 'fecct_pretrained.pt'
LIBRARY_PATH = ROOT / 'pruning_library.pt'
PRETRAIN_CODES = (
    'BCH_N31_K16', 'BCH_N63_K36', 'BCH_N63_K45', 'BCH_N63_K51',
    'POLAR_N64_K32', 'POLAR_N64_K48', 'POLAR_N128_K64', 'POLAR_N128_K86',
    'LDPC_N49_K24', 'LDPC_N121_K60', 'LDPC_N121_K70', 'LDPC_N121_K80',
)
BATCHES_PER_EPOCH = 1000


def add_shared_args(parser):
    """Existing runtime, model, and pruning settings shared with the builder."""
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gpus', type=str, default='5', help='CUDA device IDs')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--N_dec', type=int, default=6)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--h', type=int, default=8)
    parser.add_argument('--model_prun_ratio', type=float, default=0.4,
                        help='Fraction of attention/FFN MACs to remove; must match the library')
    return parser


def create_parser():
    parser = add_shared_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--do_what', choices=('train', 'prune_and_retrain_lora'),
                        default='prune_and_retrain_lora')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--test_batch_size', type=int, default=2048)
    parser.add_argument('--target_code', '--test_code', default='LDPC_N128_K64',
                        help='Code name in Codes_DB, e.g. LDPC_N128_K64')
    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=1, help='KD loss weight')
    parser.add_argument('--beta', type=float, default=1, help='Training noise scale')
    return parser


def setup(args):
    if args.workers < 0 or args.batch_size < 1 or args.N_dec < 1 or args.h < 1:
        raise ValueError('Invalid data loader or model settings')
    if args.d_model < 1 or args.d_model % args.h or not 0 <= args.model_prun_ratio < 1:
        raise ValueError('Invalid model dimension or pruning ratio')
    args.dropout_attn = args.dropout = 0
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # PyTorch 2.0 emits this internal deprecation while loading tensor checkpoints.
    warnings.filterwarnings('ignore', message='TypedStorage is deprecated.*',
                            category=UserWarning, module=r'torch\._utils')
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_code(name):
    if not re.fullmatch(r'[A-Za-z0-9]+_N[1-9][0-9]*_K[1-9][0-9]*', name):
        raise ValueError('Use a code name from Codes_DB')
    parity = read_parity_matrix(str(ROOT / 'Codes_DB' / name))
    generator = get_generator(parity)
    if generator.size == 0 or not np.any(generator) or np.any((generator @ parity.T) % 2):
        raise ValueError('Invalid generator/parity-check matrices')
    return torch.from_numpy(generator.T.copy()).long(), torch.from_numpy(parity).long()


class ECC_Dataset(Dataset):
    def __init__(self, generator, parity, sigma, length, zero_cw=True, beta=1):
        self.generator = generator.T
        self.parity = parity.T
        self.sigma = tuple(float(s) for s in sigma)
        self.length = length
        self.zero_cw = zero_cw
        self.beta = beta

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        message = (torch.zeros(self.generator.size(0), dtype=torch.long) if self.zero_cw
                   else torch.randint(0, 2, (self.generator.size(0),)))
        x = (message @ self.generator) % 2
        y = bin_to_sign(x) + torch.randn(self.parity.size(0)) * random.choice(self.sigma) * self.beta
        syndrome = bin_to_sign((sign_to_bin(torch.sign(y)).long() @ self.parity) % 2)
        return x.float(), y.float(), y.abs().float(), syndrome.float()


def make_loader(generator, parity, ebnos, args, *, training=False, calibration=False, rate=None):
    rate = generator.size(1) / parity.size(1) if rate is None else rate
    sigma = [EbN0_to_std(ebno, rate) for ebno in ebnos]
    batch_size = args.batch_size if training or calibration else args.test_batch_size
    length = batch_size * BATCHES_PER_EPOCH if training or calibration else batch_size
    dataset = ECC_Dataset(generator, parity, sigma, length, zero_cw=training,
                          beta=getattr(args, 'beta', 1) if training else 1)
    return DataLoader(dataset, batch_size=batch_size, shuffle=training, num_workers=args.workers)


def resolve_pretrained_path():
    """Prefer a newly trained checkpoint; otherwise use the distributed backbone."""
    if PRETRAINED_PATH.is_file():
        return PRETRAINED_PATH
    if PROVIDED_PRETRAINED_PATH.is_file():
        return PROVIDED_PRETRAINED_PATH
    raise FileNotFoundError('No pretrained FECCT checkpoint found')


def load_pretrained(args, device, checkpoint_path=None):
    checkpoint_path = resolve_pretrained_path() if checkpoint_path is None else checkpoint_path
    model = DC_ECC_Transformer(args).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model


def derive_entry(model, name, args, device):
    generator, parity = load_code(name)
    n, k = map(int, re.findall(r'_(?:N|K)(\d+)', name))
    loader = make_loader(generator, parity, [4], args, calibration=True, rate=k / n)
    head_mask = torch.ones(args.N_dec, args.h, device=device)
    neuron_mask = torch.ones(args.N_dec, args.d_model * 4, device=device)
    head_scores, neuron_scores = collect_mask_grads_ecc_revised(
        model, head_mask, neuron_mask, parity, loader, device)
    cfg = SimpleNamespace(hidden_size=args.d_model)
    heads, neurons = search_mac_ecc_modify(
        cfg, head_scores, neuron_scores, sum(parity.shape), 1 - args.model_prun_ratio)
    return {'code': name, 'signature': spectral_signature(parity),
            'head_mask': heads.cpu(), 'neuron_mask': neurons.cpu()}


def apply_pruning_masks(model, head_mask, neuron_mask):
    validate_masks(head_mask, neuron_mask, model_config(model.args))
    for index, layer in enumerate(model.decoder.layers):
        layer.self_attn.prune_heads(torch.where(head_mask[index] == 0)[0].tolist())
        keep = torch.where(neuron_mask[index] != 0)[0]
        layer.feed_forward.w_1 = prune_linear_layer(layer.feed_forward.w_1, keep)
        layer.feed_forward.w_2 = prune_linear_layer(layer.feed_forward.w_2, keep, dim=1)
    return model


def replace_linear_with_lora(model, r, lora_alpha):
    """Preserve the all-Linear LoRA placement used by the main experiment."""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if not module.in_features or not module.out_features:
                continue
            replacement = lora.Linear(module.in_features, module.out_features, r=r,
                                      lora_alpha=lora_alpha, bias=module.bias is not None)
            replacement.to(device=module.weight.device, dtype=module.weight.dtype)
            with torch.no_grad():
                replacement.weight.copy_(module.weight)
                if module.bias is not None:
                    replacement.bias.copy_(module.bias)
            setattr(model, name, replacement)
        else:
            replace_linear_with_lora(module, r, lora_alpha)
    return model


def train(model, parity, device, loader, optimizer, epoch, lr, teacher=None, alpha=1):
    model.train()
    if teacher is not None:
        teacher.eval()
    parity = parity.to(device)
    cum_loss = cum_ber = cum_fer = cum_samples = 0
    started = time.time()
    for x, y, magnitude, syndrome in loader:
        x, y, magnitude, syndrome = (t.to(device) for t in (x, y, magnitude, syndrome))
        z_mul = y * bin_to_sign(x)
        optimizer.zero_grad(set_to_none=True)
        logits, loss, prediction = model(magnitude, syndrome, parity, z_mul, y, parity)
        if teacher is not None:
            with torch.no_grad():
                teacher_logits, _, _ = teacher(magnitude, syndrome, parity, z_mul, y, parity)
            student_p, teacher_p = torch.sigmoid(logits), torch.sigmoid(teacher_logits)
            student_p = torch.stack((1 - student_p, student_p), dim=-1)
            teacher_p = torch.stack((1 - teacher_p, teacher_p), dim=-1)
            kd = F.kl_div(torch.log(student_p + 1e-12), teacher_p, reduction='none').sum(-1).mean()
            loss = loss + alpha * kd
        loss.mean().backward()
        optimizer.step()
        count = x.size(0)
        cum_loss += loss.item() * count
        cum_ber += BER(prediction, x) * count
        cum_fer += FER(prediction, x) * count
        cum_samples += count
    logging.info(
        f'Training epoch {epoch}, Batch {len(loader)}/{len(loader)}: LR={lr:.2e}, '
        f'Loss={cum_loss / cum_samples:.2e} BER={cum_ber / cum_samples:.2e} FER={cum_fer / cum_samples:.2e}')
    logging.info(f'Epoch {epoch} Train Time {time.time() - started}s')
    return cum_loss / cum_samples


@torch.no_grad()
def test(model, parity, device, loaders, ebnos, min_FER=100):
    model.eval()
    parity = parity.to(device)
    losses, bers, fers, samples = [], [], [], []
    started = time.time()
    for loader in loaders:
        loss_sum = bit_errors = frame_errors = count = 0
        while True:
            for x, y, magnitude, syndrome in loader:
                x, y, magnitude, syndrome = (t.to(device) for t in (x, y, magnitude, syndrome))
                _, loss, prediction = model(magnitude, syndrome, parity, y * bin_to_sign(x), y, parity)
                loss_sum += loss.item() * x.size(0)
                bit_errors += BER(prediction, x) * x.size(0)
                frame_errors += FER(prediction, x) * x.size(0)
                count += x.size(0)
                if (min_FER > 0 and frame_errors > min_FER and count > 1e5) or count >= 1e9:
                    break
            if (min_FER > 0 and frame_errors > min_FER and count > 1e5) or count >= 1e9:
                break
        losses.append(loss_sum / count)
        bers.append(bit_errors / count)
        fers.append(frame_errors / count)
        samples.append(count)
    for label, values in [('Loss', losses), ('FER', fers), ('BER', bers),
                          ('-ln(BER)', [-np.log(b) if b > 0 else float('inf') for b in bers])]:
        logging.info('Test %s %s', label, ' '.join(f'{snr}: {v:.2e}' for snr, v in zip(ebnos, values)))
    logging.info(f'# of testing samples: {samples}\nTest Time {time.time() - started} s')
    return losses, bers, fers


def pretrain(args, device):
    model = DC_ECC_Transformer(args).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    codes = {name: load_code(name) for name in PRETRAIN_CODES}
    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        generator, parity = codes[random.choice(PRETRAIN_CODES)]
        loader = make_loader(generator, parity, range(2, 8), args, training=True)
        loss = train(model, parity, device, loader, optimizer, epoch, scheduler.get_last_lr()[0])
        scheduler.step()
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), PRETRAINED_PATH)
    # As in the original training branch, evaluate the final epoch's code.
    ebnos = range(4, 7)
    loaders = [make_loader(generator, parity, [snr], args) for snr in ebnos]
    test(model, parity, device, loaders, ebnos)


def prune_and_retrain_lora(args, device):
    try:
        # Check the ratio before computing the target spectrum or deriving any mask.
        library = load_library(LIBRARY_PATH, args.model_prun_ratio, model_config(args))
        checkpoint_path = resolve_pretrained_path()
        if library['checkpoint_sha256'] != checkpoint_digest(checkpoint_path):
            raise ValueError('Library and pretrained checkpoint do not match')
        generator, parity = load_code(args.target_code)
        entry = retrieve_mask(library, parity, args.model_prun_ratio)
        model = load_pretrained(args, device, checkpoint_path)
        reused = entry is not None
        if not reused:
            entry = derive_entry(model, args.target_code, args, device)
            library['entries'].append(entry)
            save_library(library, LIBRARY_PATH)
        teacher = copy.deepcopy(model).eval().requires_grad_(False)
        apply_pruning_masks(model, entry['head_mask'], entry['neuron_mask'])
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, EOFError, pickle.UnpicklingError):
        logging.info('Library reuse failed.')
        return 1
    logging.info('Library reuse succeeded.' if reused else 'Library reuse failed.')
    replace_linear_with_lora(model, args.lora_r, args.lora_alpha)
    lora.mark_only_lora_as_trainable(model)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    for epoch in range(1, args.epochs + 1):
        loader = make_loader(generator, parity, range(2, 8), args, training=True)
        train(model, parity, device, loader, optimizer, epoch, scheduler.get_last_lr()[0],
              teacher=teacher, alpha=args.alpha)
        scheduler.step()
    # Save only the adapter and the metadata needed to reapply it to the same backbone.
    torch.save({'lora_state_dict': lora.lora_state_dict(model),
                'target_code': args.target_code, 'lora_r': args.lora_r,
                'lora_alpha': args.lora_alpha, 'model_prun_ratio': args.model_prun_ratio,
                'model_config': model_config(args), 'checkpoint_sha256': library['checkpoint_sha256'],
                'head_mask': entry['head_mask'], 'neuron_mask': entry['neuron_mask']},
               ROOT / f'lora_{args.target_code}_r{args.lora_r}.pt')
    n, k = map(int, re.findall(r'_(?:N|K)(\d+)', args.target_code))
    ebnos = range(2, 7)
    loaders = [make_loader(generator, parity, [snr], args, rate=k / n) for snr in ebnos]
    test(model, parity, device, loaders, ebnos)
    return 0


def main():
    parser = create_parser()
    args = parser.parse_args()
    if (args.epochs < 1 or args.lora_r < 1 or args.lora_alpha < 1 or args.test_batch_size < 1
            or not np.isfinite([args.lr, args.alpha, args.beta]).all()
            or args.lr <= 0 or args.alpha < 0 or args.beta <= 0):
        parser.error('Epochs, rank, batch sizes, learning rate, and noise scale must be positive')
    try:
        device = setup(args)
    except ValueError as error:
        parser.error(str(error))
    if args.do_what == 'train':
        pretrain(args, device)
        return 0
    return prune_and_retrain_lora(args, device)


if __name__ == '__main__':
    raise SystemExit(main())

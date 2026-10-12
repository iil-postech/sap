import os

import numpy as np
import torch

def read_parity_matrix(path: str) -> np.ndarray:
    # Resolve the file path and extension.
    if not os.path.splitext(path)[1]:            # No extension provided.
        if os.path.isfile(path + ".alist"):
            path += ".alist"
        elif os.path.isfile(path + ".txt"):
            path += ".txt"
        else:
            raise FileNotFoundError(f"Neither {path}.alist nor {path}.txt was found.")

    ext = os.path.splitext(path)[1].lower()

    # Select the parser based on the file extension.
    if ext == ".alist":
        return _read_pc_matrix_alist(path)
    elif ext == ".txt":
        return _read_pc_matrix_txt(path)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

# ------------------------------------------------------------------
def _read_pc_matrix_alist(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        lines = f.readlines()

    column_num, row_num = np.fromstring(lines[0], dtype=int, sep=" ")
    H = np.zeros((row_num, column_num), dtype=int)

    for col_idx in range(4, 4 + column_num):
        nz_entries = np.fromstring(lines[col_idx], dtype=int, sep=" ")
        # The alist format uses 1-based row indices.
        for row in nz_entries[nz_entries > 0]:
            H[row - 1, col_idx - 4] = 1
    return H

# ------------------------------------------------------------------
def _read_pc_matrix_txt(file_name: str) -> np.ndarray:
    with open(file_name, "r") as f:
        rows = [
            list(map(int, line.replace(",", " ").split()))
            for line in f if line.strip()
        ]
    H = np.array(rows, dtype=int)
    # Ensure the matrix contains only binary values.
    if not np.isin(H, [0, 1]).all():
        raise ValueError(f"{file_name} must contain a binary matrix (0 or 1 only).")
    return H

def row_reduce(mat, ncols=None):
    assert mat.ndim == 2
    ncols = mat.shape[1] if ncols is None else ncols
    mat_row_reduced = mat.copy()
    p = 0
    for j in range(ncols):
        idxs = p + np.nonzero(mat_row_reduced[p:,j])[0]
        if idxs.size == 0:
            continue
        mat_row_reduced[[p,idxs[0]],:] = mat_row_reduced[[idxs[0],p],:]
        idxs = np.nonzero(mat_row_reduced[:,j])[0].tolist()
        idxs.remove(p)
        mat_row_reduced[idxs,:] = mat_row_reduced[idxs,:] ^ mat_row_reduced[p,:]
        p += 1
        if p == mat_row_reduced.shape[0]:
            break
    return mat_row_reduced, p

def get_generator(pc_matrix_):
    assert pc_matrix_.ndim == 2
    pc_matrix = pc_matrix_.copy().astype(bool).transpose()
    pc_matrix_I = np.concatenate((pc_matrix, np.eye(pc_matrix.shape[0], dtype=bool)), axis=-1)
    pc_matrix_I, p = row_reduce(pc_matrix_I, ncols=pc_matrix.shape[1])
    return row_reduce(pc_matrix_I[p:,pc_matrix.shape[1]:])[0]


def sign_to_bin(x):
    return 0.5 * (1 - x)

def bin_to_sign(x):
    return 1 - 2 * x

def EbN0_to_std(EbN0, rate):
    snr =  EbN0 + 10. * np.log10(2 * rate)
    return np.sqrt(1. / (10. ** (snr / 10.)))

def BER(x_pred, x_gt):
    return torch.mean((x_pred != x_gt).float()).item()

def FER(x_pred, x_gt):
    return torch.mean(torch.any(x_pred != x_gt, dim=1).float()).item()

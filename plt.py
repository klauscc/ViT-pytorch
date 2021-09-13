# -*- coding: utf-8 -*-
#!/bin/sh
#================================================================
#   Don't go gently into that good night.
#
#   author: klaus
#   description:
#
#================================================================

import pickle as pkl
import matplotlib.pyplot as plt
import torch
import os


def plot_attention_head(attention):
    # The plot is of the attention when a token was generated.
    # The model didn't generate `<START>` in the output. Skip it.

    ax = plt.gca()
    ax.matshow(attention)


def calculate_inverse(A):
    alpha = 0.1
    attention_probs = A
    num_patches = attention_probs.shape[-1]
    I = torch.eye(num_patches, device=attention_probs.device).unsqueeze(0)
    S = torch.linalg.inv(I - alpha * attention_probs)
    return S


def plot_weights(A, save_path):
    A = A.cpu()
    S = calculate_inverse(A).numpy()
    print(A.shape, S.shape)
    fig = plt.figure(figsize=(16, 16))

    for h, head in enumerate(A):
        ax = fig.add_subplot(4, 6, h + 1)
        plot_attention_head(head)
        ax.set_xlabel(f'Head {h+1}')

    for h, head in enumerate(S):
        ax = fig.add_subplot(4, 6, 12 + h + 1)
        plot_attention_head(head)
        ax.set_xlabel(f'RW Head {h+1}')

    plt.tight_layout()
    plt.savefig(save_path)


if __name__ == "__main__":
    i = 0
    save_dir = './attn_figs'
    save_path = os.path.join(save_dir, f'attn_{i:03d}.png')

    os.makedirs(save_dir)
    data = pkl.load(open(f'attn_weights/{i:03d}.pkl', 'rb'))
    img, A = data

    plot_weights(A, save_path)

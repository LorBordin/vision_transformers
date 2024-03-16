import numpy as np
import torch

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d))) if j % 2 == 0 \
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"
    assert h % n_patches == 0, "Image size must be divisible by n_patches"

    patch_size = h // n_patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(n, n_patches * n_patches, c, patch_size, patch_size)
    patches = patches.permute(0, 1, 3, 4, 2).contiguous().view(n, n_patches * n_patches, -1)

    return patches
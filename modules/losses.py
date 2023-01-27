import torch


def normal_kl(mean1, logvar1, mean2, logvar2):
    total_kl = 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

    return total_kl

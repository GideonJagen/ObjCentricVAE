import random, os
import numpy as np
import torch
import importlib
import deeptrack as dt
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt


def distance_matrix(matrix_a, matrix_b):
    expanded_a = np.expand_dims(matrix_a, 1)  # Shape: (n, 1, m)
    expanded_b = np.expand_dims(matrix_b, 0)  # Shape: (1, n, m)
    expanded_a = np.transpose(expanded_a, (2, 1, 0))
    square_difference = np.square(
        expanded_a - expanded_b
    )  # Element-wise squared difference
    distances = np.sum(square_difference, axis=2)
    distances = np.sqrt(distances)
    return distances


def distance(x1, x2, y1, y2):
    dist = np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))
    return dist


def run_test(
    testset,
    model,
    combine_radius,
    plot_num=0,
    num_entities=10,
    pres_floor=0.25,
):
    for i_seq, x in enumerate(testset):
        x = torch.unsqueeze(x, dim=0)
        x = x.to(model.device)
        (
            recon,
            kl_divergence,
            background_kl,
            xy,
            background,
            x_hat,
            z_pres,
            pres_loss,
        ) = model(x)

        x = x.detach().cpu().numpy()
        recon = recon.detach().cpu().numpy()
        pres = z_pres.detach().cpu().numpy()
        xy = xy.detach().cpu().numpy()

        n = [i for i in range(num_entities)]

        xp = xy[:, 0]
        yp = xy[:, 1]

        plt.figure(figsize=(20, 10))

        distances = []
        for j in range(xp.shape[0]):
            distance_r = []
            for k in range(xp.shape[0]):
                if j == k:
                    distance_r.append(0)
                    continue
                dist = distance(xp[j], xp[k], yp[j], yp[k])
                distance_r.append(dist)
            distances.append(distance_r)
        distances = np.array(distances)

        # Create the combin map
        row_indices, col_indices = np.indices(distances.shape)
        matrix = np.zeros(distances.shape)
        matrix[row_indices - col_indices <= 0] = 1

        pres_mask = pres <= pres_floor

        combine_map = (
            np.where(np.array(distances) < combine_radius, 1, 0).astype(bool)
            & matrix.astype(bool)
        ).astype(np.int32)

        pres_mask_extended = np.repeat(
            np.expand_dims(~pres_mask, axis=1), pres_mask.shape[0], axis=1
        ).astype(np.int32)
        combine_map *= pres_mask_extended
        combine_map = (
            combine_map.astype(bool) | np.eye(combine_map.shape[0], dtype=bool)
        ).astype(np.int32)

        remove_map = (
            np.sum(combine_map - np.eye(combine_map.shape[0]), axis=0) + pres_mask
        ).astype(np.int32)

        remove_map = np.array(remove_map, dtype=bool)

        if i_seq == plot_num or plot_num == None:
            plt.subplot(1, 2, 1)

            plt.imshow(x[0, 0, :, :], cmap="gray")
            # plt.colorbar()
            plt.scatter(
                yp[~remove_map],
                xp[~remove_map],
                s=10,
                edgecolor="r",
                facecolor="none",
            )
            annotations = []
            for j, txt in enumerate(n):
                if remove_map[j]:
                    continue
                annotations.append(
                    plt.annotate(round(pres[j], 3), (yp[j], xp[j]), color="white")
                )
            plt.subplot(1, 2, 2)
            plt.imshow(recon[0, 0, :, :], cmap="gray")

            plt.savefig(f"../results/gif/fig_{i_seq}.png")

import torch
import torch.nn as nn
import pytorch_lightning as pl


class MultiheadAttention(pl.LightningModule):
    def __init__(
        self,
        max_t=1,
        softmax_factor=1,
        number_of_heads=1,
        num_filters=12,
        attention_mechanism="cosine-similarity",
        pos_attention_weight=1,  # Weight between position attention and morphology attention
        **kwargs
    ):
        super(MultiheadAttention, self).__init__()

        self.attention_mechanism = attention_mechanism
        self.max_t = max_t
        self.softmax_factor = softmax_factor
        self.number_of_heads = number_of_heads
        self.number_of_filters = num_filters
        self.pos_attention_weight = pos_attention_weight

        self.combine_time_dense = nn.Linear(1, 1)

    def forward(self, latents, positions, latents_frames):
        batch_size = latents.shape[0]
        timesteps = latents.shape[1]

        time_mask = self._create_time_mask(latents_frames, max_t=self.max_t)

        mask = time_mask

        eye = torch.eye(batch_size, dtype=torch.float32, device=time_mask.device)
        eye = eye.repeat_interleave(timesteps * latents.shape[2], dim=0)
        batch_mask = eye.repeat_interleave(timesteps * latents.shape[2], dim=1)

        time_mask *= batch_mask

        # Edges between the supernodes:
        updated_latents, attention_lat = self.multihead_time_attention(
            latents, self.number_of_heads
        )

        # Edges between the supernodes:
        updated_latents_pos, attention_pos = self.multihead_time_attention(
            positions, self.number_of_heads
        )

        attention = torch.cat(
            [attention_lat, attention_pos * self.pos_attention_weight], dim=0
        )
        attention = torch.mean(attention, dim=0, keepdim=True)

        # attention = torch.squeeze(
        #    self.combine_time_dense(torch.unsqueeze(attention, dim=-1)), dim=-1
        # )

        normalized_attention = torch.nn.functional.softmax(
            (attention + (1 - mask) * -10e9) / self.softmax_factor, dim=2
        )
        normalized_attention = torch.mean(normalized_attention, axis=0, keepdim=True)

        add_eye = torch.eye(
            normalized_attention.shape[1], device=normalized_attention.device
        )
        updated_latents = (
            torch.matmul(normalized_attention + add_eye, updated_latents) / 2
        )

        f = 0.0
        updated_latents_pos = torch.matmul(
            normalized_attention * f + add_eye, updated_latents_pos
        ) / (1 + f)

        updated_latents = updated_latents.view(-1, self.number_of_filters)
        # updated_latents_pos = updated_latents_pos.view(-1, self.number_of_filters)

        updated_latents = updated_latents.view(
            batch_size * timesteps, -1, updated_latents.shape[-1]
        )

        updated_latents_pos = updated_latents_pos.view(
            batch_size * timesteps, -1, updated_latents_pos.shape[-1]
        )

        return [updated_latents, updated_latents_pos, normalized_attention]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-5)

    def _create_time_mask(self, supernode_times, max_t):
        supernode_times = supernode_times.view(-1, 1)
        m = torch.where(
            torch.abs(torch.transpose(supernode_times, 0, 1) - supernode_times)
            <= max_t,
            1,
            0,
        )
        m = torch.where(
            torch.transpose(supernode_times, 0, 1) == supernode_times,
            torch.tensor(0, dtype=torch.int32),
            m,
        )

        return m.float()

    def multihead_time_attention(self, latents, number_of_heads):
        number_of_filters = latents.shape[-1]

        projection_dim = number_of_filters // number_of_heads

        latents = latents.view(-1, number_of_heads, projection_dim)
        latents = latents.transpose(0, 1)

        if self.attention_mechanism == "cosine-similarity":
            attention = self._cosine_similarity(latents)
        elif self.attention_mechanism == "dot-product":
            attention = torch.matmul(latents, latents.transpose(1, 2))
        elif self.attention_mechanism == "euclidean":
            attention = torch.div(1, self._vector_norm(latents, latents))
            assert not torch.isnan(attention).any()

        else:
            raise NotImplementedError

        return latents, attention

    def _vector_norm(self, vector1, vector2):
        dist = torch.cdist(vector1, vector2, p=2)
        return dist

    def _distance_matrix_heads(self, matrix_a, matrix_b):
        expanded_a = matrix_a.unsqueeze(2)
        expanded_b = matrix_b.unsqueeze(1)
        square_difference = torch.square(expanded_a - expanded_b)
        distances = torch.sum(square_difference, dim=3)
        distances = torch.sqrt(distances)

        return distances

    def _cosine_similarity(self, latents):
        nominator = torch.matmul(latents, latents.transpose(1, 2))
        denominator = torch.einsum(
            "hi,hj->hij", torch.norm(latents, dim=2), torch.norm(latents, dim=2)
        )

        attention = torch.div(
            nominator, denominator
        )  # Note: PyTorch's div function handles division by zero

        return attention

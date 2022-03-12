import torch
from torch.nn.functional import pdist


class AccuracyCalculator:
    def __init__(self, batch_size, time_steps, device, latent_dim):
        self._device = device
        self._batch_size = torch.as_tensor(batch_size, device=device, dtype=torch.int64)
        self._z = torch.randn(batch_size, latent_dim, device=device)
        self._mask = torch.ones((batch_size, time_steps), dtype=torch.bool, device=device)
        self._length = torch.ones(batch_size, dtype=torch.int64, device=device) * time_steps

    def compute_diversity(self, model, class_id):
        if self._batch_size == 1:
            return torch.as_tensor([0.0], device=self._device)

        y = torch.as_tensor([class_id], device=self._device).repeat(self._batch_size)
        batch = {"z": self._z, "y": y, "mask": self._mask, "lengths": self._length}
        model.eval()
        with torch.no_grad():
            batch = model.decoder(batch)
            motion = model.rot2xyz(batch["output"], batch["mask"])

        dist = pdist(motion.reshape(motion.shape[0], -1))
        diversity = torch.mean(dist)
        return diversity
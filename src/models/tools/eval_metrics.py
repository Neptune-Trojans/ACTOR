import torch
from torch.nn.functional import pdist


class AccuracyCalculator:
    def __init__(self, samples_amount, time_steps, device, latent_dim):
        self._device = device
        self._samples_amount = samples_amount
        self._z = torch.randn(samples_amount, latent_dim, device=device)
        self._mask = torch.ones((samples_amount, time_steps), dtype=torch.bool, device=device)
        self._length = torch.ones(samples_amount, dtype=torch.int64, device=device) * time_steps

    def compute_diversity(self, model, class_id):
        if self._z.shape[0] == 1:
            return torch.as_tensor([0.0], device=self._device)
        y = torch.as_tensor([class_id], device=self._device).repeat(self._samples_amount)
        batch = {"z": self._z, "y": y, "mask": self._mask, "lengths": self._length}
        model.eval()
        with torch.no_grad():
            batch = model.decoder(batch)
            motion = model.rot2xyz(batch["output"], batch["mask"])

        dist = pdist(motion.reshape(motion.shape[0], -1))
        diversity = torch.mean(dist)
        return diversity
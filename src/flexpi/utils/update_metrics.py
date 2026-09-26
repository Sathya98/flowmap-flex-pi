"""Accumulate detached loss scalars across a complete optimizer update."""
import torch


class UpdateLossMetrics:
    """Sample-weighted means and ranges of local microbatch means.

    Ranges are across microbatches and ranks, not individual examples. This
    changes reporting only; it never rescales the loss used for backward.
    """

    def __init__(self):
        self.keys = None

    def add(self, loss, metrics, batch_size):
        if batch_size < 1:
            raise ValueError('Microbatch size must be positive')
        keys = ('loss', *sorted(metrics))
        if 'loss' in metrics:
            raise ValueError('Auxiliary metric name loss is reserved')
        values = torch.stack([loss.detach().float().reshape(())] + [
            torch.as_tensor(metrics[key], device=loss.device, dtype=torch.float32).detach().reshape(())
            for key in keys[1:]])
        if self.keys is None:
            self.keys = keys
            self.total = values * batch_size
            self.minimum = values.clone()
            self.maximum = values.clone()
            self.examples = batch_size
            self.microbatches = 1
        else:
            if keys != self.keys:
                raise ValueError('Loss metric keys changed within an optimizer update')
            self.total += values * batch_size
            self.minimum = torch.minimum(self.minimum, values)
            self.maximum = torch.maximum(self.maximum, values)
            self.examples += batch_size
            self.microbatches += 1

    def finish(self, accelerator):
        if self.keys is None:
            raise ValueError('Cannot report an empty update')
        counts = self.total.new_tensor([self.examples, self.microbatches]).expand(len(self.keys), -1)
        packed = torch.cat((self.total[:, None], self.minimum[:, None], self.maximum[:, None], counts), dim=1)
        gathered = accelerator.gather(packed[None])  # [ranks, metrics, 5]
        examples = gathered[:, 0, 3].sum()
        means = gathered[:, :, 0].sum(0) / examples
        lows = gathered[:, :, 1].amin(0)
        highs = gathered[:, :, 2].amax(0)
        values = torch.stack((means, lows, highs), dim=1).cpu().tolist()
        result = dict(metrics={key: dict(mean=v[0], microbatch_min=v[1], microbatch_max=v[2])
                               for key, v in zip(self.keys, values)},
                      examples=int(examples.item()),
                      microbatches=int(gathered[:, 0, 4].sum().item()))
        self.keys = None
        self.total = self.minimum = self.maximum = None
        return result

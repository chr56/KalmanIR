import math
from collections import Counter
from torch.optim.lr_scheduler import _LRScheduler, LambdaLR


class MultiStepRestartLR(_LRScheduler):
    """ MultiStep with restarts learning rate scheme.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        milestones (list): Iterations that will decrease learning rate.
        gamma (float): Decrease ratio. Default: 0.1.
        restarts (list): Restart iterations. Default: [0].
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, milestones, gamma=0.1, restarts=(0, ), restart_weights=(1, ), last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.restarts = restarts
        self.restart_weights = restart_weights
        assert len(self.restarts) == len(self.restart_weights), 'restarts and their weights do not match.'
        super(MultiStepRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.restarts:
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return [group['initial_lr'] * weight for group in self.optimizer.param_groups]
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma**self.milestones[self.last_epoch] for group in self.optimizer.param_groups]


def get_position_from_periods(iteration, cumulative_period):
    """Get the position from a period list.

    It will return the index of the right-closest number in the period list.
    For example, the cumulative_period = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 2.

    Args:
        iteration (int): Current iteration.
        cumulative_period (list[int]): Cumulative period list.

    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_period):
        if iteration <= period:
            return i


class CosineAnnealingRestartLR(_LRScheduler):
    """ Cosine annealing with restarts learning rate scheme.

    An example of config:
    periods = [10, 10, 10, 10]
    restart_weights = [1, 0.5, 0.5, 0.5]
    eta_min=1e-7

    It has four cycles, each has 10 iterations. At 10th, 20th, 30th, the
    scheduler will restart with the weights in restart_weights.

    Args:
        optimizer (torch.nn.optimizer): Torch optimizer.
        periods (list): Period for each cosine anneling cycle.
        restart_weights (list): Restart weights at each restart iteration.
            Default: [1].
        eta_min (float): The minimum lr. Default: 0.
        last_epoch (int): Used in _LRScheduler. Default: -1.
    """

    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_min=0, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert (len(self.periods) == len(
            self.restart_weights)), 'periods and restart_weights should have the same length.'
        self.cumulative_period = [sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_period)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_period[idx - 1]
        current_period = self.periods[idx]

        return [
            self.eta_min + current_weight * 0.5 * (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * ((self.last_epoch - nearest_restart) / current_period)))
            for base_lr in self.base_lrs
        ]


class ExponentialDecayWithCosineOscillationLR(LambdaLR):
    """
    An exponential decay with cosine oscillation learning rate scheduler.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        initial_lr_factor (float): The factor for the learning rate at the start of the main schedule after warmup.
        decayed_lr_factor (float): The target decay factor for the exponential decay component at the end.
        min_lr_factor (float): The minimum learning rate factor to clamp the learning rate at.
        amplitude (float): The amplitude of the cosine oscillation.
        num_cosine_cycle (float): Cosine oscillation cycle count.
        total_iterations (int): Total number of iterations for the entire schedule.
        last_epoch (int): The index of the last epoch.
    """

    def __init__(
            self,
            optimizer,
            initial_lr_factor: float,
            decayed_lr_factor: float,
            min_lr_factor: float,
            amplitude: float,
            num_cosine_cycle: float,
            total_iterations: int,
            last_epoch: int = -1
    ):
        import numpy as np

        assert total_iterations > 0, "total_iterations must be greater than 0 to calculate scaling factors"
        assert num_cosine_cycle > 0, "num_cosine_cycle must be greater than 0"
        assert amplitude >= 0, "amplitude cannot be negative"

        self.initial_lr_factor = initial_lr_factor
        self.decayed_lr_factor = decayed_lr_factor
        self.min_lr_factor = min_lr_factor
        self.cosine_cycle_number = num_cosine_cycle
        self.amplitude = amplitude

        self.total_iterations = total_iterations

        self.delta_lr_factor = self.initial_lr_factor - self.min_lr_factor
        self.decay = -np.log(max(1e-9, self.decayed_lr_factor)) / total_iterations
        self.freq = (1 / (total_iterations / self.cosine_cycle_number))

        assert self.freq > 0, f"invalid frequency {self.freq}"

        super().__init__(optimizer, self.lr_lambda, last_epoch)

    def lr_lambda(self, current_iteration: int) -> float:
        """
        Computes the multiplicative factor for the learning rate at the current iteration.
        """
        import numpy as np
        current_iteration = float(current_iteration)

        x = current_iteration
        base_lr_factor = self.min_lr_factor + self.delta_lr_factor * np.exp(-self.decay * x)
        oscillation_factor = 1 + self.amplitude * np.cos(2 * np.pi * self.freq * x - 0.5 * np.pi)
        lr_factor = base_lr_factor * oscillation_factor
        return max(lr_factor, self.min_lr_factor)

import math

class LinearWarmupCosineDecay:
    def __init__(self, warmup_epochs, total_epochs, base_lr, min_lr, last_epoch: int = 0):
        """
        A scheduler with linear warmup followed by cosine decay.
        To be called at the beginning of each epoch and it returns the lr for the epoch
        
        Args:
            warmup_epochs: The number of epochs for the linear warmup phase.
            total_epochs: The total number of epochs for the scheduler.
            base_lr: The maximum learning rate after warmup.
            min_lr: The minimum learning rate during decay.
            last_epoch: The index of the last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.last_epoch = last_epoch

    def get_lr(self):
        current_epoch = self.last_epoch + 1

        if current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (current_epoch / max(1, self.warmup_epochs))
        else:
            # Cosine decay
            progress = (current_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))  # Cosine decay factor
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        
        return lr
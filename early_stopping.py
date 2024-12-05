# https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

class EarlyStopping:
    """Early stopping callback similar to Keras implementation"""
    def __init__(
        self,
        monitor='val_loss',
        min_delta=0.0,
        patience=5,
        verbose=1,
        mode='min',
        restore_best_weights=True
    ):
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.restore_best_weights = restore_best_weights

        self.best_weights = None
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, epoch, model, current):
        if self.mode == 'min':
            improved = current < (self.best - self.min_delta)
        else:
            improved = current > (self.best + self.min_delta)

        if improved:
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.verbose > 0:
                    print(f'Early stopping triggered at epoch {epoch}')
                if self.restore_best_weights and self.best_weights is not None:
                    if self.verbose > 0:
                        print('Restoring best weights')
                    model.load_state_dict(self.best_weights)
                return True
        return False

class EarlyStopping:
    """
    Early stops the training if patience treshold is suprassed with the objective to optimize the time spent with optuna.
    The min_delta values is the minimum acceptable value of difference between the current validation loss
    and the best validation loss, for each time the current validation loss doesn't achieve the minimum
    difference, a counter is increased, if the counter is bigger or equal to the patience, early stop becomes True
    and the training stops.
    """
    def __init__(self, patience=5, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

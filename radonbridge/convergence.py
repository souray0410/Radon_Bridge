"""Predeclared validation-plateau rule, distinct from a resource or epoch cap."""
import math

class Plateau:
    def __init__(self, *, min_epochs, max_epochs, patience, min_delta, lr_patience, lr_factor):
        if not (1 <= min_epochs <= max_epochs and 1 <= lr_patience < patience and min_delta >= 0 and 0 < lr_factor < 1):
            raise ValueError('Invalid convergence policy')
        self.minimum=min_epochs; self.maximum=max_epochs; self.patience=patience
        self.delta=min_delta; self.lr_patience=lr_patience; self.factor=lr_factor
        self.best=-math.inf; self.anchor=-math.inf; self.bad=0; self.best_epoch=None

    def update(self, score, epoch):
        if not math.isfinite(score):raise FloatingPointError('Nonfinite validation metric')
        improved=score>self.best
        if improved:self.best=score;self.best_epoch=epoch
        if score>self.anchor+self.delta:self.anchor=score;self.bad=0
        else:self.bad+=1
        return {'improved':improved,'reduce_lr':self.bad>0 and self.bad%self.lr_patience==0,
                'plateau':epoch>=self.minimum and self.bad>=self.patience}

    def state(self):
        return {'best_macro_f1':self.best,'best_epoch':self.best_epoch,'epochs_without_material_improvement':self.bad}

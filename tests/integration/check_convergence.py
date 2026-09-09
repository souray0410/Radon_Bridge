"""The stopping rule must not conflate a plateau, a tiny improvement and an epoch cap."""
from radon_bridge.convergence import Plateau
p=Plateau(min_epochs=8,max_epochs=60,patience=6,min_delta=.001,lr_patience=3,lr_factor=.3)
flags=[p.update(v,i) for i,v in enumerate([.5,.6,.6001,.6002,.6003,.6004,.6005,.6006],1)]
assert not any(v['plateau'] for v in flags[:-1]) and flags[-1]['plateau']
assert p.best_epoch==8 and flags[4]['reduce_lr']
f=p.update(.61,9);assert f['improved'] and not f['plateau'] and p.bad==0
p=Plateau(min_epochs=2,max_epochs=4,patience=3,min_delta=.001,lr_patience=1,lr_factor=.3)
assert not any(p.update(.1*i,i)['plateau'] for i in range(1,5))
try:p.update(float('nan'),5)
except FloatingPointError:pass
else:raise AssertionError('Nonfinite metric accepted')
print('{"plateau_vs_epoch_cap":true,"small_improvements_keep_best_checkpoint":true,"lr_reduction":true}')

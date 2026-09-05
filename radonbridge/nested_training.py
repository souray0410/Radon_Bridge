"""Shared prefix widths; sequential autograd and one averaged BN update per batch."""
import torch
from torch import nn


def exchanges(g):
    return [g.modules_by_name()[m['exchange_edge_name']] for m in g.communication_groups]


def widths(g):
    values=[getattr(x,'nested_rhos',None) for x in exchanges(g)]
    if not any(v is not None for v in values):return None
    if len(values)!=1 or values[0] is None:raise ValueError('This protocol supports one nested bridge')
    return values[0]


def set_width(g,rho):
    for ex in exchanges(g):ex.set_rho(rho)


def bn_snapshot(g):
    return {m:{n:v.detach().clone() for n,v in m.named_buffers(recurse=False)}
            for m in g.graph.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)}


@torch.no_grad()
def restore_bn(state):
    for m,values in state.items():
        for n,v in values.items():getattr(m,n).copy_(v)


@torch.no_grad()
def merge_bn(states):
    # Mean of per-width running-stat updates, NOT variance of a pooled mixture.
    # num_batches_tracked increments once, not three times.
    for m in states[0]:
        for n in states[0][m]:
            values=[s[m][n] for s in states]
            if values[0].is_floating_point():getattr(m,n).copy_(torch.stack(values).mean(0))
            else:
                assert all(torch.equal(values[0],v) for v in values[1:])
                getattr(m,n).copy_(values[0])


def training_backward(g,c,o,y,scale=1.):
    rhos=widths(g)
    if rhos is None:
        _,loss=g.forward(c,o,y,loss_scale=scale)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss')
        g.backward();return loss.detach()
    initial=bn_snapshot(g);updates=[];total=c.new_zeros(())
    try:
        for rho in rhos:
            restore_bn(initial);set_width(g,rho)
            _,loss=g.forward(c,o,y,loss_scale=scale/len(rhos))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite nested loss')
            g.backward();total+=loss.detach();updates.append(bn_snapshot(g))
        merge_bn(updates)
    except BaseException:
        restore_bn(initial);raise
    finally:set_width(g,max(rhos))
    return total


def aggregate_metrics(per_width):
    values=list(per_width.values())
    return {'tasks':{b:{'macro_f1':sum(v['tasks'][b]['macro_f1'] for v in values)/len(values)} for b in ('cfp','oct')},
            'mean_task_macro_f1':sum(v['mean_task_macro_f1'] for v in values)/len(values),
            'per_width':per_width,
            'aggregation':'equal mean of six branch-width F1 values, not averaged probabilities',
            'default_prediction_file':'maximum width only; each width also has a named prediction file'}

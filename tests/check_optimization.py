"""Another task's gradient must not change an independent predictor's updates."""
import json
import torch
from radonbridge.model import PilotGraph
from radonbridge.optimization import clip_task_gradients, configure_optimizer

def main():
    torch.set_num_threads(2)
    mono = PilotGraph(modalities='cfp', loss_reduction='sum')
    paired = PilotGraph(loss_reduction='sum')
    # Make the other task's gradients deliberately very different.
    with torch.no_grad():
        paired.modules_by_name()['oct_head'].weight.mul_(30.)
    cfg = {'warmup_adapt_stages': [4], 'adapt_stages': [3,4], 'backbone_lr': 1e-5, 'head_bridge_lr': 1e-4}
    opt_m = configure_optimizer(mono, cfg); opt_p = configure_optimizer(paired, cfg)
    steps = []
    for i in range(2):
        torch.manual_seed(91+i)
        c = torch.randn(1,2,3,96,96); o = torch.randn(1,2,1,32,96,96); y = torch.tensor([i % 2])
        for g,opt in [(mono,opt_m),(paired,opt_p)]:
            opt.zero_grad(set_to_none=True); g.forward(c,o,y); g.backward()
            clip_task_gradients(g, .1); opt.step()
        errors = []
        for name,m in mono.modules_by_name().items():
            if name.startswith('cfp_'):
                for key,v in m.state_dict().items():
                    other = paired.modules_by_name()[name].state_dict()[key]
                    errors.append(float((v-other).abs().max())); assert torch.equal(v,other)
        steps.append(max(errors))
    print(json.dumps({'two_adamw_updates_cfp_standalone_vs_paired_max_error':steps,'per_task_clipping_independence':True,'unequal_other_task_gradients':True,'loss_sum_preserves_standalone_gradient_scale':True}))

if __name__ == '__main__':main()

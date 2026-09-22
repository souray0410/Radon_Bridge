"""CFP/OCT convenience interface backed by the reusable MHD V5 task runtime."""
from radon_bridge.models.networks import build_cfp_oct, TaskLoss
from radon_bridge.models.task import MHDTaskGraph

class PilotGraph(MHDTaskGraph):
    def __init__(self, *, bridge_configs=(), seed=3407, device='cpu', backbone='resnet18', task_fusion=None):
        native=build_cfp_oct(seed=seed,backbone=backbone,task_fusion=task_fusion)
        super().__init__(native,bridge_configs=bridge_configs,device=device)

    def set_inputs(self, cfp, oct_, target):
        self.builder.set_inputs(dict(zip(('cfp','oct','target'),(cfp,oct_,target))))

    def forward(self, cfp, oct_, target, loss_scale=1.):
        return self.forward_inputs(dict(zip(('cfp','oct','target'),(cfp,oct_,target))),loss_scale=loss_scale)

    def native_forward(self, cfp, oct_, target):
        return self.native_forward_inputs(dict(zip(('cfp','oct','target'),(cfp,oct_,target))))

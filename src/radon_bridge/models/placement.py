"""Single-process source model parallelism; pinned MHD remains the executor."""
import torch


def transfer(value, destination, transport='cpu_staged'):
    target = torch.device(destination)
    if (transport == 'cpu_staged' and value.device.type == 'cuda' and
            target.type == 'cuda' and value.device != target):
        # Both copies remain in autograd. Do not trust advertised peer capability:
        # qualification on ws02 found an actual CUDA0->CUDA1 copy returning zeros.
        return value.to('cpu').to(target)
    return value.to(target)


def place(model, sources, communication_device, *, transport='cpu_staged'):
    if transport not in ('cpu_staged',):
        raise ValueError('Unqualified transfer transport')
    if set(sources) != set(model.source_keys): raise ValueError('Every source requires a device')
    if hasattr(model, '_placement_hooks'): raise ValueError('Placement already applied')
    plan = {k: str(torch.device(v)) for k,v in sources.items()}
    communication_device = str(torch.device(communication_device))
    modules = model.task.modules_by_name(); ownership = {}
    for key, names in model.task.definition.source_modules.items():
        for name in names: ownership[name] = plan[key]
        ownership[key+'_criterion'] = plan[key]
    for node in model.task.nodes:
        matching = [k for k in plan if node.name.startswith(k+'_')]
        node.to_device(plan[matching[0]] if matching else communication_device)
    handles = []
    for name, module in modules.items():
        target = ownership.get(name, communication_device)
        module.to(target)
        def inputs(_module, args, destination=target):
            return tuple(transfer(x,destination,transport) if isinstance(x,torch.Tensor) else x for x in args)
        handles.append(module.register_forward_pre_hook(inputs))
        if name.startswith('bridge_') and name.endswith('_return'):
            matched = [k for k in plan if '_'+k+'_stage' in name]
            if len(matched) != 1: raise ValueError('Ambiguous native return device')
            destination = plan[matched[0]]
            def output(_module, args, result, destination=destination): return transfer(result,destination,transport)
            handles.append(module.register_forward_hook(output))
    model._placement_hooks = handles
    model.device_placement = dict(schema='source_model_parallel_v1', sources=plan, communication=communication_device,
                                  processes=1, transport=transport, effective_batch_unchanged=True)
    return model

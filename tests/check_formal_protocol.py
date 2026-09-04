"""Run with stdlib only; reject unfrozen or over-budget multi-seed launches."""
import copy
import importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('protocol',Path(__file__).parents[1]/'radonbridge/protocol.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
valid={'phase_mode':'formal','confirmation_seeds':[3411,3412,3413,3414,3415],
       'formal_arms':[{'id':'independent','mode':'baseline'},{'id':'rb','mode':'radon'}],
       'recipes':[{'backbone':'resnet34','adapt_stages':[1,2,3,4],'training_regime':'full_finetune'}],
       'training_regime':'full_finetune','batchnorm_policy':'train','recipe_policy':'user_full_finetune_override','recipe_override_reason':'User correction', 'prior_gpu_minutes':20.,'max_minutes':220.,
       'qualification_summary_sha256':'a'*64,'qualification_summary_path':'/recorded/summary.json'}
module.validate_formal_protocol(valid)
invalid=[]
for field,value in [('confirmation_seeds',[3411]*5),('prior_gpu_minutes',21.),('max_minutes',float('nan')),('qualification_summary_sha256','pending'),('recipes',[])]:
    candidate=copy.deepcopy(valid);candidate[field]=value
    try:module.validate_formal_protocol(candidate)
    except (ValueError,TypeError):invalid.append(field)
    else:raise AssertionError(field)
print({'valid_budget_boundary_passed':True,'rejected_invalid_launches':invalid})

full=copy.deepcopy(valid)
full.update(training_regime='full_finetune',batchnorm_policy='train',recipe_policy='user_full_finetune_override',recipe_override_reason='Explicit user correction')
full['recipes']=[{'adapt_stages':[1,2,3,4],'training_regime':'full_finetune'}]
module.validate_formal_protocol(full)
full['recipes'][0]['adapt_stages']=[3,4]
try: module.validate_formal_protocol(full)
except ValueError: print({'partial_freeze_rejected_in_full_finetune':True})
else: raise AssertionError('Partial freezing silently accepted')

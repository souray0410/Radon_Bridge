"""Definitions of the archived three-seed 31-contrast benchmark; no launcher."""
SEEDS = [3416, 3417, 3418]
RHOS = [.0625, .125, .25]
BASELINES = ['mmtm_r4', 'mmtm_r8', 'attention_d128', 'attention_d256']

def key(seed, arm, rho):
    return (seed, arm, rho or 0.)

"""Pre-launch guards for the five-seed stage and cumulative GPU budget."""
import math
import re

def validate_formal_protocol(protocol):
    if protocol.get('phase_mode') != 'formal':
        return
    seeds = protocol['confirmation_seeds']; arms = protocol['formal_arms']
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('Formal stage requires five distinct seeds')
    if len(protocol['recipes']) != 1 or not isinstance(protocol['recipes'][0], dict):
        raise ValueError('One frozen qualification recipe required')
    if not arms or arms[0]['mode'] != 'baseline' or arms[0]['id'] != 'independent':
        raise ValueError('Matched baseline must run first for each seed')
    if sum(a['mode'] == 'baseline' for a in arms) != 1 or len({a['id'] for a in arms}) != len(arms):
        raise ValueError('Unique arms and exactly one baseline required')
    prior = float(protocol['prior_gpu_minutes']); remaining = float(protocol['max_minutes'])
    if not all(math.isfinite(x) for x in [prior, remaining]) or prior < 0 or remaining <= 0 or prior + remaining > 240:
        raise ValueError('Cumulative stage008 budget cannot exceed240GPU minutes')
    if not re.fullmatch(r'[0-9a-f]{64}', protocol['qualification_summary_sha256']):
        raise ValueError('Verified qualification summary hash required')
    if not protocol.get('qualification_summary_path'):
        raise ValueError('Qualification evidence path required')

"""Scalar compatibility and heterogeneous projection/compression shape checks."""
import json
import torch
from radonbridge.bridge import BridgeExchange, FeatureSpec


def main():
    torch.set_num_threads(2)
    specs=[FeatureSpec('cfp',2,(3,4)),FeatureSpec('oct',2,(3,3,4))]
    torch.manual_seed(42);a=BridgeExchange(specs,M=4,S=11,rho=.5).double()
    torch.manual_seed(42);b=BridgeExchange(specs,M={'oct':4,'cfp':4},S=11,rho={'oct':.5,'cfp':.5}).double()
    assert a.state_dict().keys()==b.state_dict().keys()
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    torch.nn.init.normal_(a.mixer.conv.weight,std=.02);b.load_state_dict(a.state_dict())
    xs=[torch.randn(2,s.channels,*s.shape,dtype=torch.double) for s in specs]
    assert torch.equal(a(*xs),b(*xs))
    c=BridgeExchange(specs,M={'cfp':4,'oct':8},S=11,rho={'cfp':.5,'oct':.25}).double()
    assert [x['projected_channels'] for x in c.metadata['participants']]==[8,16]
    assert c.mixer.widths==(4,4)
    assert torch.equal(c(*xs),torch.cat([x.flatten(1) for x in xs],dim=1))
    invalid=[('M',{'cfp':4}),('M',{'cfp':4,'oct':0}),('M',{'cfp':4,'oct':8,'typo':4}),('rho',{'cfp':.5}),('rho',{'cfp':.5,'oct':0})]
    for key,value in invalid:
        args={'M':4,'S':11,'rho':.5};args[key]=value
        try:BridgeExchange(specs,**args)
        except ValueError:pass
        else:raise AssertionError('Invalid participant mapping accepted')
    print(json.dumps({'scalar_mapping_bitwise_equivalent':True,'heterogeneous_projected_channels':[8,16],'retained_widths':[4,4],'zero_bridge_exact':True,'invalid_maps_rejected':True}))


if __name__=='__main__':main()

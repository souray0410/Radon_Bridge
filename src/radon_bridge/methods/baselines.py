"""Single-stage communication adapters, preserving native task predictions.

MMTM follows the paper's 2*sigmoid excitation, with identity initialization.
Attention is a simultaneous bidirectional pre-LN attention adapter, not a
complete Transformer. Neither adapter claims width matching with Radon.
"""
import math
import torch
from torch import nn
from radon_bridge.methods.projection import positive_integer


class NativeExchange(nn.Module):
    def __init__(self,specs,family):
        super().__init__()
        self.keys=[s.key for s in specs]
        if len(specs)!=2 or len(set(self.keys))!=2:raise ValueError('This baseline requires two distinct sources')
        self.shapes=[(s.channels,*s.shape) for s in specs]
        self.lengths=[math.prod(s) for s in self.shapes]
        self.family=family;self.compression=None
        self.latest_inputs=self.latest_deltas=None
        self.metadata={'family':family,'compression':None,'M':None,'S':None,'rho':None,
            'participants':[{'key':s.key,'channels':s.channels,'shape':list(s.shape),'tokens':math.prod(s.shape)} for s in specs]}

    def finish_metadata(self):
        n=sum(p.numel() for p in self.parameters())
        self.metadata.update(stored_bridge_parameters=n,effective_bridge_parameters=n)

    def export_fixed_bases(self,directory):return []

    def check(self,features):
        if len(features)!=len(self.shapes) or any(tuple(x.shape[1:])!=s for x,s in zip(features,self.shapes)):
            raise ValueError('Participant shapes changed')
        if len({x.shape[0] for x in features})!=1:raise ValueError('Participants must share batch alignment')

    def packet(self,features,deltas):
        self.latest_inputs=tuple(x.detach() for x in features)
        self.latest_deltas=tuple(x.detach() for x in deltas)
        return torch.cat([(x+d).flatten(1) for x,d in zip(features,deltas)],1)


class MMTMExchange(NativeExchange):
    def __init__(self,specs,reduction_ratio):
        super().__init__(specs,'mmtm');positive_integer(reduction_ratio,'reduction_ratio',1)
        hidden=int(2*sum(s.channels for s in specs)/reduction_ratio)
        if hidden<1:raise ValueError('MMTM hidden dimension must be positive')
        self.squeeze=nn.Linear(sum(s.channels for s in specs),hidden)
        self.excite=nn.ModuleList([nn.Linear(hidden,s.channels) for s in specs])
        for layer in self.excite:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.metadata.update(reduction_ratio=reduction_ratio,hidden_dimension=hidden,
            adaptation='single stage; paper 2*sigmoid; zero excitation weight and bias for identity initialization')
        self.finish_metadata()

    def forward(self,*features):
        self.check(features)
        shared=torch.relu(self.squeeze(torch.cat([x.flatten(2).mean(-1) for x in features],1)))
        deltas=[x*(2*torch.sigmoid(layer(shared))-1).reshape(x.shape[0],x.shape[1],*([1]*(x.ndim-2))) for x,layer in zip(features,self.excite)]
        return self.packet(features,deltas)


class AttentionDirection(nn.Module):
    def __init__(self,source_channels,destination_channels,dimension,heads):
        super().__init__();self.heads=heads;self.dimension=dimension
        self.query=nn.Linear(destination_channels,dimension)
        self.key=nn.Linear(source_channels,dimension)
        self.value=nn.Linear(source_channels,dimension)
        self.output=nn.Linear(dimension,destination_channels)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,destination,source):
        n=destination.shape[0];h=self.heads;d=self.dimension//h
        q=self.query(destination).reshape(n,-1,h,d).transpose(1,2)
        k=self.key(source).reshape(n,-1,h,d).transpose(1,2)
        v=self.value(source).reshape(n,-1,h,d).transpose(1,2)
        # Explicit definition avoids backend-dependent dropout or fused kernels.
        z=(torch.softmax(q@k.transpose(-1,-2)/math.sqrt(d),dim=-1)@v).transpose(1,2).reshape(n,-1,self.dimension)
        return self.output(z)


class AttentionExchange(NativeExchange):
    def __init__(self,specs,attention_dimension,heads):
        super().__init__(specs,'cross_attention')
        positive_integer(attention_dimension,'attention_dimension',1);positive_integer(heads,'heads',1)
        if attention_dimension%heads:raise ValueError('Attention dimension must divide into heads')
        self.norms=nn.ModuleList([nn.LayerNorm(s.channels) for s in specs])
        self.directions=nn.ModuleList([AttentionDirection(specs[1-i].channels,specs[i].channels,attention_dimension,heads) for i in range(2)])
        self.metadata.update(attention_dimension=attention_dimension,heads=heads,pre_layernorm=True,
            direction_order=[[self.keys[1-i],self.keys[i]] for i in range(2)],
            projection_bias=True,dropout=0.,feed_forward=False,explicit_position_encoding=False,
            adaptation='full spatial tokens; simultaneous bidirectional attention; zero output projections')
        self.finish_metadata()

    def forward(self,*features):
        self.check(features)
        tokens=[norm(x.flatten(2).transpose(1,2)) for norm,x in zip(self.norms,features)]
        deltas=[layer(tokens[i],tokens[1-i]).transpose(1,2).reshape_as(features[i]) for i,layer in enumerate(self.directions)]
        return self.packet(features,deltas)


class CMXRectifyExchange(NativeExchange):
    """CMX Feature Rectify Module adapted to heterogeneous spatial dimensionality.

    On equal 2D grids and alignment_tokens == H*W this is numerically the
    author FRM formula. For 2D/3D native features, only the spatial
    correspondence is adapted through a deterministic flattened token lattice.
    This is not a reproduction of the full CMX segmentation system.
    """
    AUTHOR_REPOSITORY='huaaaliu/RGBX_Semantic_Segmentation'
    AUTHOR_COMMIT='e251d860aebc2f583a6c4919877e6bebe7f1aff3'
    AUTHOR_NET_UTILS_SHA256='ada5e36e14d83c35d9230618c4eb84352a76ec2275a086a624b280e7638d8473'
    AUTHOR_LICENSE_SHA256='a3fb69f7d2d7ab44ce80bba7f8c3a61f3c8a2775a2baac4bef815a60c4d8ba5e'

    def __init__(self,specs,alignment_tokens):
        super().__init__(specs,'cmx_frm')
        positive_integer(alignment_tokens,'alignment_tokens',1)
        channels={s.channels for s in specs}
        if len(channels)!=1:raise ValueError('CMX-FRM author formula requires equal channel dimensions')
        self.dim=next(iter(channels));self.alignment_tokens=alignment_tokens
        self.lambda_c=.5;self.lambda_s=.5;reduction=1
        # Project identity initialization, analogous to the zero-residual
        # initialization already used for MMTM and cross-attention adapters.
        # The internal FRM formula remains unchanged; these two learned scalars
        # only gate the final directional residual so epoch0 exactly reproduces
        # the accepted native parents.
        self.residual_gate=nn.Parameter(torch.zeros(2))
        self.channel_mlp=nn.Sequential(
            nn.Linear(self.dim*4,self.dim*4//reduction),nn.ReLU(inplace=True),
            nn.Linear(self.dim*4//reduction,self.dim*2),nn.Sigmoid())
        # Author SpatialWeights uses 1x1 Conv2d on a common 2D grid. Kernel-1
        # Conv1d is algebraically identical after flattening when grids match.
        self.spatial_mlp=nn.Sequential(
            nn.Conv1d(self.dim*2,self.dim//reduction,kernel_size=1),nn.ReLU(inplace=True),
            nn.Conv1d(self.dim//reduction,2,kernel_size=1),nn.Sigmoid())
        self.metadata.update(
            alignment_tokens=alignment_tokens,reduction=1,lambda_c=.5,lambda_s=.5,
            author_repository=self.AUTHOR_REPOSITORY,author_commit=self.AUTHOR_COMMIT,
            author_net_utils_sha256=self.AUTHOR_NET_UTILS_SHA256,
            author_license='MIT',author_license_sha256=self.AUTHOR_LICENSE_SHA256,
            author_component='FeatureRectifyModule only; not full CMX encoder/fusion/segmentation system',
            adaptation='channel branch preserves author pooling/MLP exactly; spatial 1x1 weighting uses a deterministic shared flattened-token lattice for heterogeneous 2D/3D grids; project adds a trainable zero-initialized directional residual gate solely for exact-parent initialization',
            identity_initialization='trainable residual_gate[2] initialized to 0; author-equivalent FRM recovered at gate=1',
            spatial_alignment='flatten -> deterministic half-pixel linear resample to shared token lattice -> author-equivalent 1x1 spatial MLP -> deterministic resample directional weight/source to destination native shape')
        self.finish_metadata()

    @staticmethod
    def _resize(tokens,size):
        if tokens.shape[-1]==size:return tokens
        source=tokens.shape[-1]
        if source<1 or size<1:raise ValueError('CMX-FRM token sizes must be positive')
        # Deterministic equivalent of interpolate(..., mode='linear', align_corners=False).
        # CUDA does not provide a deterministic backward for interpolate1d, so
        # compute the fixed half-pixel coordinates explicitly and use gather.
        position=(torch.arange(size,device=tokens.device,dtype=torch.float64)+.5)*(source/size)-.5
        position=position.clamp(0,source-1)
        left=position.floor().to(torch.long);right=(left+1).clamp(max=source-1)
        weight=(position-left.to(position.dtype)).to(tokens.dtype).reshape(*([1]*(tokens.ndim-1)),size)
        a=tokens.index_select(-1,left);b=tokens.index_select(-1,right)
        return a+(b-a)*weight

    def forward(self,*features):
        self.check(features)
        batch=features[0].shape[0];channels=self.dim
        flat=[x.flatten(2) for x in features]
        avg=torch.cat([x.mean(-1) for x in flat],1)
        maximum=torch.cat([x.amax(-1) for x in flat],1)
        channel=self.channel_mlp(torch.cat((avg,maximum),1)).reshape(batch,2,channels)
        shared=[self._resize(x,self.alignment_tokens) for x in flat]
        spatial=self.spatial_mlp(torch.cat(shared,1))
        deltas=[]
        for destination in range(2):
            source=1-destination;tokens=flat[destination].shape[-1]
            source_native=self._resize(flat[source],tokens).reshape_as(features[destination])
            channel_weight=channel[:,source].reshape(batch,channels,*([1]*(features[destination].ndim-2)))
            spatial_weight=self._resize(spatial[:,source:source+1],tokens).reshape(
                batch,1,*features[destination].shape[2:])
            frm_delta=self.lambda_c*channel_weight*source_native + self.lambda_s*spatial_weight*source_native
            deltas.append(self.residual_gate[destination]*frm_delta)
        return self.packet(features,deltas)


class CMXChannelWeights2D(nn.Module):
    """Author CMX ChannelWeights on an already aligned 2-D grid.

    Global average/max pooling are written as explicit reductions.  They are
    algebraically identical to AdaptiveAvgPool2d(1)/AdaptiveMaxPool2d(1), but
    avoid PyTorch's nondeterministic CUDA adaptive-max-pool backward.
    """
    def __init__(self,dim,reduction=1):
        super().__init__();self.dim=dim
        self.mlp=nn.Sequential(
            nn.Linear(dim*4,dim*4//reduction),nn.ReLU(inplace=True),
            nn.Linear(dim*4//reduction,dim*2),nn.Sigmoid())

    def forward(self,x1,x2):
        b=x1.shape[0];x=torch.cat((x1,x2),dim=1)
        avg=x.mean(dim=(2,3));maximum=x.amax(dim=(2,3))
        y=self.mlp(torch.cat((avg,maximum),dim=1))
        return y.reshape(b,2,self.dim,1,1).permute(1,0,2,3,4)


class CMXSpatialWeights2D(nn.Module):
    """Author CMX SpatialWeights on an already aligned 2-D grid."""
    def __init__(self,dim,reduction=1):
        super().__init__();self.dim=dim
        self.mlp=nn.Sequential(
            nn.Conv2d(dim*2,dim//reduction,kernel_size=1),nn.ReLU(inplace=True),
            nn.Conv2d(dim//reduction,2,kernel_size=1),nn.Sigmoid())

    def forward(self,x1,x2):
        b,_,h,w=x1.shape
        value=self.mlp(torch.cat((x1,x2),dim=1))
        return value.reshape(b,2,1,h,w).permute(1,0,2,3,4)


class CMXFeatureRectify2D(nn.Module):
    """Author FeatureRectifyModule on one shared 2-D grid."""
    def __init__(self,dim,reduction=1,lambda_c=.5,lambda_s=.5):
        super().__init__();self.lambda_c=lambda_c;self.lambda_s=lambda_s
        self.channel_weights=CMXChannelWeights2D(dim,reduction)
        self.spatial_weights=CMXSpatialWeights2D(dim,reduction)

    def forward(self,x1,x2):
        channel=self.channel_weights(x1,x2);spatial=self.spatial_weights(x1,x2)
        return (
            x1+self.lambda_c*channel[1]*x2+self.lambda_s*spatial[1]*x2,
            x2+self.lambda_c*channel[0]*x1+self.lambda_s*spatial[0]*x1,
        )


class CMXCrossAttention(nn.Module):
    """Author CMX cross-attention used inside FeatureFusionModule."""
    def __init__(self,dim,num_heads):
        super().__init__();positive_integer(num_heads,'heads',1)
        if dim%num_heads:raise ValueError('CMX FFM dimension must divide into heads')
        self.dim=dim;self.num_heads=num_heads;self.scale=(dim//num_heads)**-.5
        self.kv1=nn.Linear(dim,dim*2,bias=False)
        self.kv2=nn.Linear(dim,dim*2,bias=False)

    def forward(self,x1,x2):
        if x1.shape!=x2.shape or x1.ndim!=3:raise ValueError('CMX FFM aligned BNC tokens required')
        b,n,c=x1.shape;h=self.num_heads;d=c//h
        q1=x1.reshape(b,n,h,d).permute(0,2,1,3).contiguous()
        q2=x2.reshape(b,n,h,d).permute(0,2,1,3).contiguous()
        k1,v1=self.kv1(x1).reshape(b,n,2,h,d).permute(2,0,3,1,4).contiguous()
        k2,v2=self.kv2(x2).reshape(b,n,2,h,d).permute(2,0,3,1,4).contiguous()
        ctx1=(k1.transpose(-2,-1)@v1)*self.scale;ctx1=ctx1.softmax(dim=-2)
        ctx2=(k2.transpose(-2,-1)@v2)*self.scale;ctx2=ctx2.softmax(dim=-2)
        y1=(q1@ctx2).permute(0,2,1,3).reshape(b,n,c).contiguous()
        y2=(q2@ctx1).permute(0,2,1,3).reshape(b,n,c).contiguous()
        return y1,y2


class CMXCrossPath(nn.Module):
    """Source-faithful CMX CrossPath stage."""
    def __init__(self,dim,reduction,num_heads):
        super().__init__();positive_integer(reduction,'reduction',1)
        inner=dim//reduction
        if inner<1 or dim%reduction:raise ValueError('CMX FFM reduction must divide channels')
        self.channel_proj1=nn.Linear(dim,inner*2)
        self.channel_proj2=nn.Linear(dim,inner*2)
        self.cross_attn=CMXCrossAttention(inner,num_heads)
        self.end_proj1=nn.Linear(inner*2,dim)
        self.end_proj2=nn.Linear(inner*2,dim)
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim)

    def forward(self,x1,x2):
        y1,u1=torch.relu(self.channel_proj1(x1)).chunk(2,dim=-1)
        y2,u2=torch.relu(self.channel_proj2(x2)).chunk(2,dim=-1)
        v1,v2=self.cross_attn(u1,u2)
        return (
            self.norm1(x1+self.end_proj1(torch.cat((y1,v1),dim=-1))),
            self.norm2(x2+self.end_proj2(torch.cat((y2,v2),dim=-1))),
        )


class CMXChannelEmbed(nn.Module):
    """Source-faithful 2-D ChannelEmbed stage from CMX FFM."""
    def __init__(self,in_channels,out_channels,reduction):
        super().__init__();positive_integer(reduction,'reduction',1)
        hidden=out_channels//reduction
        if hidden<1 or out_channels%reduction:raise ValueError('CMX FFM reduction must divide channels')
        self.residual=nn.Conv2d(in_channels,out_channels,kernel_size=1,bias=False)
        self.channel_embed=nn.Sequential(
            nn.Conv2d(in_channels,hidden,kernel_size=1,bias=True),
            nn.Conv2d(hidden,hidden,kernel_size=3,stride=1,padding=1,bias=True,groups=hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden,out_channels,kernel_size=1,bias=True),
            nn.BatchNorm2d(out_channels),
        )
        self.norm=nn.BatchNorm2d(out_channels)

    def forward(self,x):
        return self.norm(self.residual(x)+self.channel_embed(x))


class CMXFeatureFusion(nn.Module):
    """CMX FeatureFusionModule on one shared 2-D grid."""
    def __init__(self,dim,num_heads,reduction=1):
        super().__init__()
        self.cross=CMXCrossPath(dim,reduction,num_heads)
        self.channel_emb=CMXChannelEmbed(dim*2,dim,reduction)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module,nn.Linear):
            nn.init.trunc_normal_(module.weight,std=.02)
            if module.bias is not None:nn.init.zeros_(module.bias)
        elif isinstance(module,nn.LayerNorm):
            nn.init.zeros_(module.bias);nn.init.ones_(module.weight)
        elif isinstance(module,nn.Conv2d):
            fan_out=module.kernel_size[0]*module.kernel_size[1]*module.out_channels
            fan_out//=module.groups
            nn.init.normal_(module.weight,0,math.sqrt(2.0/fan_out))
            if module.bias is not None:nn.init.zeros_(module.bias)

    def forward(self,x1,x2):
        if x1.shape!=x2.shape or x1.ndim!=4:raise ValueError('CMX FFM requires aligned BCHW features')
        b,c,h,w=x1.shape
        t1,t2=self.cross(x1.flatten(2).transpose(1,2),x2.flatten(2).transpose(1,2))
        merged=torch.cat((t1,t2),dim=-1).transpose(1,2).reshape(b,c*2,h,w).contiguous()
        return self.channel_emb(merged)


class CMXFullExchange(NativeExchange):
    """Project adaptation of the CMX FRM+FFM communication core.

    The author FRM and FFM math run on a deterministic shared square 2-D
    lattice.  Returning the single author fused feature to two native branches
    is project-specific and is isolated behind zero-initialized scalar gates.
    This is the full CMX *communication core*, not the author's segmentation
    decoder/system.
    """
    def __init__(self,specs,alignment_tokens,heads):
        super().__init__(specs,'cmx_full')
        positive_integer(alignment_tokens,'alignment_tokens',1);positive_integer(heads,'heads',1)
        side=math.isqrt(alignment_tokens)
        if side*side!=alignment_tokens:raise ValueError('CMX full FFM requires a square shared 2-D token lattice')
        channels={s.channels for s in specs}
        if len(channels)!=1:raise ValueError('CMX full core requires equal channel dimensions')
        self.dim=next(iter(channels));self.alignment_tokens=alignment_tokens;self.alignment_side=side
        self.frm=CMXFeatureRectify2D(self.dim,reduction=1,lambda_c=.5,lambda_s=.5)
        self.ffm=CMXFeatureFusion(self.dim,heads,reduction=1)
        self.return_gate=nn.Parameter(torch.zeros(2))
        self.metadata.update(
            alignment_tokens=alignment_tokens,alignment_shape=[side,side],heads=heads,reduction=1,
            author_repository=CMXRectifyExchange.AUTHOR_REPOSITORY,author_commit=CMXRectifyExchange.AUTHOR_COMMIT,
            author_net_utils_sha256=CMXRectifyExchange.AUTHOR_NET_UTILS_SHA256,
            author_license='MIT',author_license_sha256=CMXRectifyExchange.AUTHOR_LICENSE_SHA256,
            author_component='FeatureRectifyModule + FeatureFusionModule communication core; not full CMX segmentation system',
            adaptation='align each native feature deterministically to one shared square 2-D lattice first; run author FeatureRectifyModule then FeatureFusionModule on that shared lattice; resize the single fused feature back to each native shape behind project-only zero-initialized return gates',
            identity_initialization='project return_gate[2] initialized to 0; author FRM+FFM core remains trainable',
            source_fidelity='FRM+FFM core retained; segmentation decoder and native RGB-X same-grid hierarchy are outside this classifier adaptation')
        self.finish_metadata()

    def _align_to_shared(self,feature):
        tokens=CMXRectifyExchange._resize(feature.flatten(2),self.alignment_tokens)
        return tokens.reshape(feature.shape[0],self.dim,self.alignment_side,self.alignment_side)

    def core_forward_shared(self,x1,x2):
        if x1.shape!=x2.shape or x1.ndim!=4:
            raise ValueError('CMX full author core requires aligned equal-shape BCHW features')
        if x1.shape[1]!=self.dim or tuple(x1.shape[2:])!=(self.alignment_side,self.alignment_side):
            raise ValueError('CMX full shared-grid identity changed')
        x1,x2=self.frm(x1,x2)
        return self.ffm(x1,x2)

    def core_from_native(self,*features):
        self.check(features)
        shared=[self._align_to_shared(feature) for feature in features]
        return self.core_forward_shared(*shared)

    def forward(self,*features):
        fused=self.core_from_native(*features)
        deltas=[]
        for index,feature in enumerate(features):
            returned=CMXRectifyExchange._resize(fused.flatten(2),math.prod(feature.shape[2:])).reshape_as(feature)
            deltas.append(self.return_gate[index]*returned)
        return self.packet(features,deltas)

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


class AuthorMMTMExchange(NativeExchange):
    """Author-code sigmoid excitation, separate from the historical identity adapter.

    Mechanism reference: haamoon/mmtm at 1c81cfefad5532cfb39193b8af3840ac3346e897.
    Complete medical task composition lives in models.modern_communication.
    """
    def __init__(self, specs, reduction_ratio):
        super().__init__(specs, 'mmtm_author')
        positive_integer(reduction_ratio, 'reduction_ratio')
        hidden = int(2 * sum(s.channels for s in specs) / reduction_ratio)
        if hidden < 1:
            raise ValueError('MMTM hidden dimension must be positive')
        self.squeeze = nn.Linear(sum(s.channels for s in specs), hidden)
        self.excite = nn.ModuleList([nn.Linear(hidden, s.channels) for s in specs])
        self.metadata.update(reduction_ratio=reduction_ratio, hidden_dimension=hidden,
            gate_scale=1., initialization='PyTorch Linear defaults; not identity',
            author_commit='1c81cfefad5532cfb39193b8af3840ac3346e897',
            adaptation='author-code MMTM mechanism; medical task and encoders are adaptations')
        self.finish_metadata()

    def forward(self, *features):
        self.check(features)
        pooled = torch.cat([x.flatten(2).mean(-1) for x in features], dim=1)
        shared = torch.relu(self.squeeze(pooled))
        outputs = []
        for x, excite in zip(features, self.excite):
            gate = torch.sigmoid(excite(shared)).reshape(x.shape[:2] + (1,) * (x.ndim - 2))
            outputs.append((x * gate).flatten(1))
        # Direct multiplication preserves the author operation, without a
        # subtract/add residual round trip changing FP32 rounding.
        return torch.cat(outputs, dim=1)


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

"""Inference-only static donor projections; identical live native-H queries."""
import math
import torch
from fusion import Fusion as UncachedFusion, FusionEpsilon as UncachedEpsilon


class Fusion(UncachedFusion):
    def __init__(self, variant='full', width=512):
        super().__init__(variant,width)
        self._donor_cache = None
        self._cache_versions = None

    def prepare_cache(self, features, enabled):
        self._donor_cache = None
        self._cache_versions = None
        if not enabled or self.training or any(p.requires_grad for p in self.parameters()):
            return
        estimate = sum((int(f['mask'].sum())**2*(self.n_heads+4) + int(f['mask'].sum())*(self.n_heads*self.rank+1024))
                       *len(self.layers)*4 for f in features)
        if features[0]['single'].is_cuda:
            free,total=torch.cuda.mem_get_info(features[0]['single'].device)
            if estimate>min(4*2**30,total*.15,free*.25):
                return
        elif estimate > 256*2**20:
            return
        cache={}
        with torch.no_grad():
            for f in features:
                m=int(f['mask'].sum())
                assert f['single'].shape==(m,384) and f['pair'].shape==(m,m,128)
                dtype=self.h_norm.weight.dtype
                single=self.s_norm(f['single'].to(dtype))
                keys={str(layer):self.key[str(layer)](single).reshape(m,self.n_heads,self.rank)
                      for layer in self.layers}
                blocks={str(layer):[] for layer in self.layers}
                for start in range(0,m,32):
                    pair=self.pair_encoder(f['pair'][start:start+32].to(dtype))
                    for layer in self.layers:
                        blocks[str(layer)].append(self.pair_out[str(layer)](pair).permute(2,0,1))
                reader_cache={}
                encoded=self.single_encoder(f['single'].float())
                for layer in self.layers:
                    reader=self.readers[str(layer)]
                    reader_cache[str(layer)]={
                        'k':reader['k'](encoded).reshape(m,4,64),
                        'v':reader['v'](encoded).reshape(m,4,64),
                        'direct':reader['direct'](encoded),
                        'pair':reader['pair'](self.pair_encoder[0](f['pair'].float())).permute(2,0,1),
                    }
                cache[id(f)] = (f,keys,blocks,reader_cache)
        self._donor_cache=cache
        self._cache_versions=tuple(p._version for p in self.parameters())

    def forward(self,H,valid_nodes,features,layer):
        if (self._donor_cache is None or self.training or torch.is_grad_enabled()
            or self._cache_versions!=tuple(p._version for p in self.parameters())):
            return super().forward(H,valid_nodes,features,layer)
        if layer not in self.layers or H.shape[-1]!=self.hidden_size:
            raise ValueError('Native layer/hidden dimension does not match adapter')
        if valid_nodes.shape!=H.shape[:2] or valid_nodes.dtype!=torch.bool or len(features)!=len(H):
            raise ValueError('Native batch/mask and donor records differ')
        name=str(layer)
        bias=H.new_zeros((len(H),self.n_heads,H.shape[1],H.shape[1]))
        for b,f in enumerate(features):
            original,keys,blocks,_=self._donor_cache[id(f)]
            assert original is f
            positions=valid_nodes[b].nonzero(as_tuple=True)[0]
            mask=f['mask']
            if mask.dtype!=torch.bool or mask.ndim!=1 or len(mask)!=len(positions):
                raise ValueError('Donor mask must match native graph nodes exactly')
            donor_positions=positions[mask.nonzero(as_tuple=True)[0]]
            m=len(donor_positions)
            if not m:
                continue
            q=self.query[name](self.h_norm(H[b,positions].to(self.h_norm.weight.dtype)))
            q=q.reshape(len(positions),self.n_heads,self.rank)
            cross=torch.einsum('ihd,jhd->hij',q,keys[name])/math.sqrt(self.rank)
            bias[b,:,positions[:,None],donor_positions[None,:]]=cross.to(H.dtype)
            for bi,start in enumerate(range(0,m,32)):
                rows=donor_positions[start:start+32]
                bias[b,:,rows[:,None],donor_positions[None,:]]=(
                    bias[b,:,rows[:,None],donor_positions[None,:]]+blocks[name][bi].to(H.dtype))
        return bias

    def residual(self,H,valid_nodes,features,layer):
        if (self._donor_cache is None or self.training or torch.is_grad_enabled()
            or self._cache_versions!=tuple(p._version for p in self.parameters())):
            return super().residual(H,valid_nodes,features,layer)
        reader=self.readers[str(layer)]
        result=torch.zeros_like(H)
        for b,f in enumerate(features):
            original,_,_,cached=self._donor_cache[id(f)]
            assert original is f
            positions=valid_nodes[b].nonzero(as_tuple=True)[0]
            covered=f['mask'].nonzero(as_tuple=True)[0]
            if not len(covered):continue
            c=cached[str(layer)]
            q=reader['q'](reader['norm'](H[b,positions])).reshape(-1,4,64)
            logits=torch.einsum('ihd,jhd->hij',q,c['k'])/math.sqrt(64)
            logits[:,covered]=logits[:,covered]+c['pair']
            messages=torch.einsum('hij,jhd->ihd',logits.softmax(-1),c['v']).flatten(1)
            delta=reader['out'](messages)
            result[b,positions]=delta.index_add(0,covered,c['direct'])
        return result


class FusionEpsilon(UncachedEpsilon):
    def set_features(self,features,cache=True):
        super().set_features(features,cache=cache)
        self.fusion.prepare_cache(features,cache)

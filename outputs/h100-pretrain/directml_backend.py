"""Optional DirectML attention implementation; checkpoint parameter names unchanged."""
import math,types
import torch
def forward_model(self,tokens,targets=None):
    logits=self.dml_original_forward(tokens)
    if targets is None: return logits
    # DirectML 0.2.5 gives a different loss for ignore_index=-100. Remove
    # ignored positions explicitly before cross entropy. Its reduction='none'
    # backward also produced zero gradients on this stack, so keep mean reduction.
    labels=targets.reshape(-1).detach().cpu()
    indices=torch.nonzero(labels!=-100,as_tuple=False).flatten()
    if indices.numel()==0: raise ValueError('No supervised target tokens')
    selected=logits.reshape(-1,self.cfg.vocab_size).index_select(0,indices.to(logits.device))
    return torch.nn.functional.cross_entropy(selected.float(),labels.index_select(0,indices).to(logits.device))
def forward_attention(self,x):
    b,t,d=x.shape
    q,k,v=self.qkv(x).reshape(b,t,3,self.heads,self.head_dim).permute(2,0,3,1,4).unbind(0)
    q,k=self.rotate(q),self.rotate(k)
    scores=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.head_dim)
    weights=torch.softmax(scores+self.dml_causal_mask[:t,:t],dim=-1)
    z=torch.matmul(weights,v)
    return self.out(z.transpose(1,2).contiguous().view(b,t,d))
def prepare_directml(model,device_index=None):
    import torch_directml
    names=[torch_directml.device_name(i).rstrip('\x00') for i in range(torch_directml.device_count())]
    if device_index is None:
        matches=[i for i,name in enumerate(names) if 'RX 580' in name.upper()]
        if len(matches)!=1: raise RuntimeError('Could not uniquely identify RX 580: '+repr(names))
        device_index=matches[0]
    device=torch_directml.device(device_index)
    mask=torch.triu(torch.full((model.cfg.context,model.cfg.context),float('-inf')),diagonal=1)
    for block in model.blocks:
        block.attn.register_buffer('dml_causal_mask',mask,persistent=False)
        block.attn.forward=types.MethodType(forward_attention,block.attn)
    model.dml_original_forward=model.forward
    model.forward=types.MethodType(forward_model,model)
    model.to(device)
    return device,names[device_index]

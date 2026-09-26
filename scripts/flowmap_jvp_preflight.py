"""Check BF16 temporal JVP and parameter backward on the allocated hardware."""
import argparse
import json
from pathlib import Path

import torch
from torch.autograd import forward_ad as fw

from flexpi.models.helpers.flowmap import dX_dt_forward_ad
from flexpi.models.helpers.normalization import ForwardADLayerNorm
from flexpi.models.helpers.checkpoint import checkpoint
from flexpi.models.wan_video_dit import DiTBlock, precompute_freqs_cis


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',default='cuda',choices=('cuda','cpu'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.manual_seed(19)
    torch.set_num_threads(2)
    device=torch.device(args.device)
    dtype=torch.bfloat16
    x=torch.randn(1,4,64,device=device,dtype=dtype)
    native=torch.nn.LayerNorm(64).to(device=device,dtype=dtype)
    with fw.dual_level():
        p,d=fw.unpack_dual(native(fw.make_dual(x,torch.randn_like(x))))
        native_dtypes=dict(primal=str(p.dtype),tangent=str(d.dtype))
    safe=ForwardADLayerNorm(64).to(device=device,dtype=dtype)
    with fw.dual_level():
        p,d=fw.unpack_dual(safe(fw.make_dual(x,torch.randn_like(x))))
        assert p.dtype==d.dtype==dtype
    block=DiTBlock(hidden_dim=64,attn_head_dim=32,num_heads=2,ffn_dim=128).to(device=device,dtype=dtype)
    context=torch.randn(1,3,64,device=device,dtype=dtype)
    modulation=torch.randn(1,6,64,device=device,dtype=dtype)
    freqs=precompute_freqs_cis(32,end=4).to(device).unsqueeze(1)
    time=torch.tensor([.3],device=device)
    results=[]
    for checkpointed in (False,True):
        block.zero_grad(set_to_none=True)
        def predict(t):
            args=(x,context,modulation+t.to(dtype).reshape(1,1,1),freqs)
            return checkpoint(block,*args) if checkpointed else block(*args)
        output,derivative=dX_dt_forward_ad(predict,time)
        assert output.dtype==derivative.dtype==dtype
        loss=output.float().square().mean()+derivative.float().square().mean()
        loss.backward()
        gradients={name:p.grad.detach().cpu().clone() for name,p in block.named_parameters() if p.grad is not None}
        assert torch.isfinite(loss) and gradients and all(torch.isfinite(g).all() for g in gradients.values())
        assert block.cross_attn.q.weight.grad.norm()>0
        results.append((output.detach().cpu(),derivative.detach().cpu(),gradients))
    for index in (0,1):torch.testing.assert_close(results[0][index],results[1][index],rtol=0,atol=0)
    assert results[0][2].keys()==results[1][2].keys()
    for name in results[0][2]:
        torch.testing.assert_close(results[0][2][name],results[1][2][name],rtol=.015,atol=.00015,
                                   msg=lambda message:f'{name}: {message}')
    report=dict(torch_version=torch.__version__,device=str(device),
        gpu_name=torch.cuda.get_device_name() if device.type=='cuda' else None,
        native_layernorm_dtypes=native_dtypes,loss=float(loss.detach()),
        gradient_tensors=len(gradients),checkpoint_gradient_agreement=True,passed=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print('BF16 temporal-JVP preflight passed:',json.dumps(report),flush=True)


if __name__=='__main__':main()

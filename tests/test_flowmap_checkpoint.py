"""Exact derivatives and activation retention for checkpointed joint Flow Maps."""
import pytest
import torch

from flexpi.models.helpers.checkpoint import checkpoint
from flexpi.models.helpers.flowmap import tuple_jvp
from flexpi.models.helpers.attention import scaled_dot_product_attention
from test_flowmap import tiny_model, batch, clone_teacher


def test_checkpoint_preserves_mixed_derivatives_and_nested_arguments():
    torch.set_num_threads(2)
    torch.manual_seed(91)
    x=torch.randn(2,3,dtype=torch.float64,requires_grad=True)
    tangent=torch.randn_like(x,requires_grad=True)
    weight=torch.randn(3,3,dtype=torch.float64,requires_grad=True)
    def objective(a,dx,w,enabled=True):
        def layer(payload,*,scale):
            value=(payload['x'] @ w).sin()*scale
            return {'value':value,'unused':None}
        def predict(z):
            call=checkpoint if enabled else lambda fn,*a,**kw:fn(*a,**kw)
            return (call(layer,{'x':z},scale=1.3)['value'],)
        (y,),(dy,)=tuple_jvp(predict,(a,),(dx,))
        return (y.square()+dy.square()).sum()
    expected=objective(x,tangent,weight,False)
    actual=objective(x,tangent,weight)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    for a,b in zip(torch.autograd.grad(actual,(x,tangent,weight)),
                   torch.autograd.grad(expected,(x,tangent,weight))):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    assert torch.autograd.gradcheck(objective,(x,tangent,weight),eps=1e-5,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('dtype',[torch.float32,torch.bfloat16])
def test_fulljoint_checkpoint_matches_loss_and_every_parameter_gradient(dtype):
    torch.set_num_threads(2)
    torch.manual_seed(19)
    model=tiny_model(('action','video','dino','pointmap'),objective='lmd',
                     dtype=dtype,lmd_teacher_gradient='full')
    object.__setattr__(model,'flow_map_teacher',clone_teacher(model).eval().requires_grad_(False))
    with torch.no_grad():
        for expert in (model.video_expert,model.action_expert):
            expert.time_embedding_delta[-1].weight.normal_(std=.003)
    direct=tiny_model(('action','video','dino','pointmap'),objective='lmd',
                      dtype=dtype,lmd_teacher_gradient='full')
    direct.load_state_dict(model.state_dict())
    object.__setattr__(direct,'flow_map_teacher',model.flow_map_teacher)
    for owner,enabled in ((model,True),(direct,False)):
        for module in owner.modules():
            for attr in ('use_gradient_checkpointing','mot_checkpoint_mixed_attn'):
                if hasattr(module,attr):setattr(module,attr,enabled)
    sample=batch(dtype,b=1)
    values=[]
    for owner in (direct,model):
        torch.manual_seed(99)
        loss,_=owner.training_loss(sample)
        loss.backward()
        values.append(loss.detach())
        assert all(p.grad is None for p in owner.flow_map_teacher.parameters())
    torch.testing.assert_close(*values,rtol=0,atol=0)
    count=0
    for (name,a),(other,b) in zip(direct.named_parameters(),model.named_parameters()):
        assert name==other
        if a.grad is None:
            assert b.grad is None,name
        else:
            assert b.grad is not None and torch.isfinite(b.grad).all(),name
            tolerance=.015 if dtype==torch.bfloat16 else 1e-5
            torch.testing.assert_close(a.grad,b.grad,rtol=tolerance,atol=tolerance/100,
                                       msg=lambda message:f'{name}: {message}')
            count+=1
    assert count>50


def test_checkpoint_reduces_saved_attention_tensors_without_detaching_jvp():
    torch.set_num_threads(2)
    torch.manual_seed(91)
    q=torch.randn(1,2,64,32,requires_grad=True)
    k=torch.randn_like(q,requires_grad=True)
    v=torch.randn_like(q,requires_grad=True)
    dq=torch.randn_like(q)
    records=[]
    for enabled in (False,True):
        saved={}
        def pack(tensor):
            if torch._is_zerotensor(tensor):return tensor
            storage=tensor.untyped_storage()
            saved[storage.data_ptr()]=storage.nbytes()
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
            def predict(a):
                out=(checkpoint(scaled_dot_product_attention,a,k,v) if enabled
                     else scaled_dot_product_attention(a,k,v))
                return (out,)
            (y,),(dy,)=tuple_jvp(predict,(q,),(dq,))
        gradients=torch.autograd.grad(y.square().mean()+dy.square().mean(),(q,k,v))
        records.append((sum(saved.values()),y,dy,gradients))
    print('Retained attention storage bytes (direct/checkpoint):', [r[0] for r in records])
    assert records[1][0] < records[0][0]*.4, [r[0] for r in records]
    for index in (1,2):torch.testing.assert_close(records[0][index],records[1][index])
    for a,b in zip(records[0][3],records[1][3]):torch.testing.assert_close(a,b)

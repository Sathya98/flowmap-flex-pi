"""Mixed-precision temporal/spatial derivatives through the actual stream model."""
import pytest
import torch
from torch.nn import functional as F
from torch.autograd import forward_ad as fw

from flexpi.models.helpers.normalization import ForwardADLayerNorm
from flexpi.models.helpers.flowmap import tuple_jvp
from test_flowmap import tiny_model, batch, clone_teacher


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64,torch.bfloat16])
def test_normal_forward_preserves_native_layernorm(dtype):
    layer=ForwardADLayerNorm((2,4),eps=1e-6).to(dtype)
    x=torch.randn(3,2,4,dtype=dtype)
    torch.testing.assert_close(layer(x),F.layer_norm(x,(2,4),layer.weight,layer.bias,layer.eps),
                               rtol=0,atol=0)


def test_bfloat_primal_and_tangent_match_after_fp32_statistics():
    # FP32 affine weights give CPU LayerNorm the FP32 saved statistics that
    # CUDA uses with BF16 weights. Before the fix its BF16 output had a FP32
    # tangent and the following BF16 Linear failed during temporal JVP.
    norm=ForwardADLayerNorm(8)
    linear=torch.nn.Linear(8,8).bfloat16()
    x=torch.randn(1,4,8,dtype=torch.bfloat16,requires_grad=True)
    with fw.dual_level():
        y=norm(fw.make_dual(x,torch.randn_like(x)))
        value,tangent=fw.unpack_dual(y)
        assert value.dtype==tangent.dtype==torch.bfloat16
        torch.testing.assert_close(value,F.layer_norm(x,(8,),norm.weight,norm.bias,norm.eps),
                                   rtol=.01,atol=.01)
        out=fw.unpack_dual(linear(y))
        loss=out.primal.float().square().mean()+out.tangent.float().square().mean()
    loss.backward()
    for p in (x,norm.weight,linear.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0


def test_layernorm_jvp_matches_native_values_and_finite_difference_gradients():
    torch.manual_seed(4)
    norm=ForwardADLayerNorm((2,3),eps=1e-6).double()
    x=torch.randn(2,2,3,dtype=torch.float64,requires_grad=True)
    tangent=torch.randn_like(x)
    ref,ref_jvp=torch.autograd.functional.jvp(
        lambda a:F.layer_norm(a,(2,3),norm.weight,norm.bias,norm.eps),x,tangent,create_graph=True)
    (value,),(jvp,)=tuple_jvp(lambda a:(norm(a),),(x,),(tangent,))
    torch.testing.assert_close(value,ref,rtol=1e-10,atol=1e-10)
    torch.testing.assert_close(jvp,ref_jvp,rtol=1e-10,atol=1e-10)
    # Check the loss gradient through the JVP numerically: this mixed derivative
    # is what LMD needs, including derivatives with respect to affine weights.
    def objective(a,weight,bias):
        def predict(z):
            return (torch.func.functional_call(norm,{'weight':weight,'bias':bias},(z,)),)
        (y,),(dy,)=tuple_jvp(predict,(a,),(tangent,))
        return (y.square()+dy.square()).sum()
    assert torch.autograd.gradcheck(objective,(x,norm.weight,norm.bias),
                                   eps=1e-5,atol=1e-5,rtol=1e-5)


@pytest.mark.parametrize('fp32_norm',[False,True])
def test_fulljoint_bfloat16_lmd_forward_backward(fp32_norm):
    torch.set_num_threads(2);torch.manual_seed(19)
    streams=('action','video','dino','pointmap')
    model=tiny_model(streams,objective='lmd',dtype=torch.bfloat16,lmd_teacher_gradient='full')
    if fp32_norm:
        for layer in model.modules():
            if isinstance(layer,torch.nn.LayerNorm) and layer.elementwise_affine:layer.float()
    teacher=clone_teacher(model).eval().requires_grad_(False)
    object.__setattr__(model,'flow_map_teacher',teacher)
    with torch.no_grad():
        for expert in (model.video_expert,model.action_expert):
            expert.time_embedding_delta[-1].weight.normal_(std=.003)
    loss,metrics=model.training_loss(batch(torch.bfloat16,b=1))
    assert torch.isfinite(loss)
    loss.backward()
    for stream in streams:assert 'loss_flowmap_'+stream in metrics
    gradients=[p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    for expert in (model.video_expert,model.action_expert):
        assert expert.time_embedding_delta[-1].weight.grad.norm()>0
    assert all(p.grad is None for p in teacher.parameters())

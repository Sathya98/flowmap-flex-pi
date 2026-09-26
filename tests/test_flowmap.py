"""CPU regression tests. No model downloads, datasets, GPU, or simulator needed."""
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from flexpi.models.action_dit import ActionDiT
from flexpi.models.wan_video_dit import WanVideoDiT
from flexpi.models.mot import MoT
from flexpi.models.flexpi import FlexPi
from flexpi.models.helpers.adaptation import (
    AdaptedLinear, install_adaptation, configure_trainable, clone_teacher,
    validate_teacher_payload,
)
from flexpi.models.helpers.flowmap import (
    FlowMapConfig, affine_flow_map, dX_dt_forward_ad, dX_dt_finite_difference,
    map_residuals, sample_level_pair_strip,
)
from flexpi.models.helpers.flex_joint import FlexJointConfig


def tiny_model(streams=('action',), objective='lsd', mode='full', pointmap=True, dtype=torch.float32, **flow_options):
    a = ActionDiT(hidden_dim=64, action_dim=8, ffn_dim=128, text_dim=16, freq_dim=16,
                  eps=1e-6, num_heads=2, attn_head_dim=32, num_layers=2, use_time_delta=True)
    v = WanVideoDiT(hidden_dim=64, in_dim=4, out_dim=4, ffn_dim=128, text_dim=16,
                    freq_dim=16, eps=1e-6, patch_size=(1,2,2), num_heads=2,
                    attn_head_dim=32, num_layers=2, has_image_input=False,
                    seperated_timestep=True, require_vae_embedding=False,
                    require_clip_embedding=False, fuse_vae_embedding_in_latents=True,
                    video_attention_mask_mode='first_frame_causal',
                    use_time_delta=True)
    mot = MoT({'video':v, 'action':a}, mot_checkpoint_mixed_attn=True,
              hbridge_enabled=True, hbridge_bottom_ratio=.5, hbridge_top_ratio=0.)
    vae = nn.Identity()
    vae.temporal_downsample_factor = 4
    vae.upsampling_factor = 16
    vae.model = SimpleNamespace(z_dim=4)
    model = FlexPi(v,a,mot,vae,nn.Identity(),nn.Identity(),text_dim=16,
                  dino_dim=8,dino_cam_patches=[(2,2)],dino_cam_regions=[(0,2,0,2)],
                  dino_pred_x0=True, device='cpu',torch_dtype=dtype,
                  joint_video='video' in streams,joint_dino='dino' in streams,
                  joint_pointmap='pointmap' in streams,enable_pointmap=pointmap,
                  flow_map=FlowMapConfig(enabled=True,streams=streams,objective=objective,mode=mode,rank=2,**flow_options),
                  flex_joint=FlexJointConfig(enabled=True,p_jv=1.,p_jd=1.,p_jp=1.))
    install_adaptation(model)
    configure_trainable(model)
    model.build_inputs = lambda sample, tiled=False: sample
    return model


def batch(dtype=torch.float32, b=2):
    video=torch.randn(b,4,2,4,4,dtype=dtype)
    return dict(context=torch.randn(b,3,16,dtype=dtype),context_mask=torch.ones(b,3,dtype=torch.bool),
                input_latents=video,first_frame_latents=video[:,:,:1],
                fuse_vae_embedding_in_latents=True,action=torch.randn(b,4,8,dtype=dtype),
                action_is_pad=torch.zeros(b,4,dtype=torch.bool),action_dim_is_pad=None,image_is_pad=None,
                dino_features=torch.randn(b,8,2,4,1,dtype=dtype),pointmap_raw=torch.randn_like(video))


class FlowMapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_sampler_full_support_and_precision(self):
        for width in (.5,1.):
            s,t=sample_level_pair_strip(10000,width,'cpu',torch.bfloat16)
            self.assertEqual(s.dtype,torch.float32)
            self.assertLess(float(s.min()),.05)
            self.assertGreater(float(s.max()),.99)
            self.assertTrue(bool((t<s).all()))
            self.assertTrue(bool(((s-t)<=width+1e-6).all()))

    def test_shifted_schedule_coverage(self):
        cfg=FlowMapConfig(strip_width=.5,schedule_shift=6.)
        with self.assertRaisesRegex(ValueError,'exceeds'):
            cfg.inference_nodes(2,'cpu')
        cfg.strip_width=1.
        self.assertAlmostEqual(float(cfg.inference_nodes(2,'cpu')[1]),6/7,places=6)

    def test_fd_boundaries_and_bfloat_rejection(self):
        t=torch.tensor([0.,.3,1.],dtype=torch.float64)
        def f(x):
            self.assertTrue(bool(((x>=0)&(x<=1)).all()))
            return x.square()
        _,d=dX_dt_finite_difference(f,t,eps=1e-4)
        torch.testing.assert_close(d,2*t,atol=1e-8,rtol=1e-8)
        with self.assertRaisesRegex(ValueError,'float32'):
            dX_dt_finite_difference(lambda x:x.bfloat16(),t.float())

    def test_attention_jvp_backward_regression(self):
        model=tiny_model().action_expert
        with torch.no_grad():model.time_embedding_delta[-1].weight.normal_(std=.02)
        x=torch.randn(2,4,8);ctx=torch.randn(2,3,16);s=torch.tensor([.8,.7]);t=torch.tensor([.4,.3])
        def f(z):return affine_flow_map(x,model(x,s*1000,ctx,timestep_delta=(z-s)*1000),s,z)
        _,d=dX_dt_forward_ad(f,t)
        d.square().mean().backward()
        grad=model.time_embedding_delta[-1].weight.grad
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.norm()),0.)

    def test_all_objectives_constant_flow(self):
        x=(torch.randn(2,3),torch.randn(2,5))
        s=torch.tensor([.9,.7]);t=torch.tensor([.3,.1])
        predict=lambda state,lo,hi:tuple(torch.ones_like(a)*2 for a in state)
        teacher=lambda state,time:predict(state,time,time)
        for objective in ('lmd','emd','pfmm','lsd','esd','psd_m','psd_u'):
            with self.subTest(objective=objective):
                residual=map_residuals(predict,teacher,x,s,t,FlowMapConfig(objective=objective))
                for r in residual:torch.testing.assert_close(r,torch.zeros_like(r),atol=3e-6,rtol=0)

    def test_joint_eulerian_spatial_derivative(self):
        # Coupled ODE x'=y, y'=x: dropping cross-stream JVP terms fails this test.
        x=(torch.tensor([[.3],[.6]],dtype=torch.float64),torch.tensor([[.7],[.1]],dtype=torch.float64))
        s=torch.tensor([.9,.8],dtype=torch.float64);t=torch.tensor([.2,.4],dtype=torch.float64)
        def predict(state,lo,hi):
            h=(hi-lo)[:,None]
            a=torch.sinh(h)/h;b=(torch.cosh(h)-1)/h
            return (b*state[0]+a*state[1],a*state[0]+b*state[1])
        for objective in ('lmd','emd','psd_m','psd_u'):
            residual=map_residuals(predict,lambda state,time:(state[1],state[0]),x,s,t,FlowMapConfig(objective=objective))
            for r in residual:torch.testing.assert_close(r,torch.zeros_like(r),atol=1e-12,rtol=0)

    def test_temporal_derivative_gradient_contract(self):
        s=torch.tensor([.8]);t=torch.tensor([.3]);x=(torch.zeros(1,1),)
        for objective in ('lmd','emd'):
            for detach in (False,True):
                parameter=torch.tensor(2.,requires_grad=True)
                predict=lambda state,lo,hi:(parameter*(hi if objective=='lmd' else lo)[:,None],)
                teacher=lambda state,time:(torch.ones_like(state[0]),)
                cfg=FlowMapConfig(objective=objective,detach_derivatives=detach)
                residual=map_residuals(predict,teacher,x,s,t,cfg)[0]
                residual.square().sum().backward()
                if objective=='lmd':
                    coefficient=t if detach else 2*t-s
                else:
                    coefficient=s if detach else 2*s-t
                torch.testing.assert_close(parameter.grad,2*residual.detach().squeeze()*coefficient.squeeze())

    def test_all_stream_subsets_train(self):
        for n in range(4):
            for subset in itertools.combinations(('video','dino','pointmap'),n):
                streams=('action',)+subset
                with self.subTest(streams=streams):
                    model=tiny_model(streams)
                    data = batch()
                    data["_flowmap_diagonal_mask"] = torch.zeros(2, dtype=torch.bool)
                    loss,metrics=model.training_loss(data)
                    self.assertTrue(bool(torch.isfinite(loss)))
                    loss.backward()
                    self.assertEqual(set(k[13:] for k in metrics if k.startswith('loss_flowmap_')),set(streams))
                    self.assertIsNone(model._batch_flex)
                    self.assertGreater(float(model.action_expert.time_embedding_delta[-1].weight.grad.norm()),0)

    def test_objectives_joint_backward_teacher_frozen(self):
        for objective in ('lmd','emd','pfmm','lsd','esd','psd_m','psd_u'):
            with self.subTest(objective=objective):
                model=tiny_model(('action','video','dino','pointmap'),objective)
                if model.flow_map.needs_teacher:
                    teacher=clone_teacher(model)
                    object.__setattr__(model,'flow_map_teacher',teacher)
                loss,_=model.training_loss(batch())
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
                if model.flow_map_teacher is not None:
                    self.assertTrue(all(p.grad is None for p in model.flow_map_teacher.parameters()))
                    self.assertFalse(any('flow_map_teacher' in k for k in model.state_dict()))

    def test_adaptation_trainable_and_checkpoint(self):
        for mode in ('full','heads','lora','adapter'):
            with self.subTest(mode=mode):
                model=tiny_model(('action','video'),mode=mode)
                loss,_=model.training_loss(batch());loss.backward()
                if mode in ('lora','adapter'):
                    layer=model.action_expert.blocks[0].self_attn.q
                    self.assertIsInstance(layer,AdaptedLinear)
                    self.assertIsNone(layer.weight.grad)
                    self.assertIsNotNone(layer.adapter_up.grad)
                model.eval();configure_trainable(model)
                self.assertFalse(any(p.requires_grad for p in model.vae.parameters()))
                with tempfile.TemporaryDirectory() as folder:
                    path=Path(folder)/'model.pt';model.save_checkpoint(path)
                    other=tiny_model(('action','video'),mode=mode)
                    other.load_checkpoint(path)
                    for k,v in model.mot.state_dict().items():torch.testing.assert_close(v,other.mot.state_dict()[k])

    def test_no_pointmap_and_padding(self):
        model=tiny_model(('action','video'),pointmap=False)
        data=batch();data['action_is_pad'][:]=True
        loss,metrics=model.training_loss(data)
        self.assertEqual(metrics['loss_flowmap_action'],0.)
        self.assertEqual(metrics['loss_diagonal_action'],0.)
        loss.backward()

    def test_bfloat16_self_distillation(self):
        model=tiny_model(('action','video'),dtype=torch.bfloat16)
        loss,_=model.training_loss(batch(torch.bfloat16));loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_inference_all_stream_subsets(self):
        data=batch(b=1)
        for n in range(4):
            for subset in itertools.combinations(('video','dino','pointmap'),n):
                with self.subTest(subset=subset):
                    model=tiny_model(('action',)+subset).eval()
                    model._encode_input_image_latents_tensor=lambda **kw:data['first_frame_latents']
                    model.dino_encoder.encode_video=lambda *a,**kw:data['dino_features'][:,:,:1]
                    model._encode_first_frame_pointmap_raw=lambda **kw:data['pointmap_raw'][:,:,:1]
                    for k in (1,2):
                        out=model.infer_action(input_image=torch.zeros(1,3,64,64),
                            action_horizon=4,num_video_frames=5,context=data['context'],
                            context_mask=data['context_mask'],per_cam_depth={},
                            num_inference_steps=k,seed=123,return_stream_latents=True)
                        self.assertEqual(out['action'].shape,(4,8))
                        self.assertTrue(bool(torch.isfinite(out['action']).all()))
                        for name in subset:
                            latent=out[name+'_latents']
                            self.assertTrue(bool(torch.isfinite(latent).all()))
                            original={'video':'input_latents','dino':'dino_features','pointmap':'pointmap_raw'}[name]
                            torch.testing.assert_close(latent[:,:,:1],data[original][:,:,:1])

    def test_teacher_clone_is_independent(self):
        model=tiny_model(objective='lmd')
        teacher=clone_teacher(model)
        self.assertTrue(model.flow_map.enabled)
        self.assertFalse(teacher.flow_map.enabled)
        self.assertIs(teacher.vae,model.vae)
        self.assertIsNot(teacher.action_expert.head.weight,model.action_expert.head.weight)
        object.__setattr__(model,'flow_map_teacher',teacher)
        configure_trainable(model)
        self.assertFalse(any(p.requires_grad for p in teacher.parameters()))

    def test_released_teacher_checkpoint_contract(self):
        model=tiny_model(('action','video','dino','pointmap'),objective='lmd',mode='lora')
        teacher=clone_teacher(model)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'teacher.pt'
            teacher.save_checkpoint(path)
            payload=torch.load(path)
            payload['mot']={k:v for k,v in payload['mot'].items()
                            if 'time_embedding_delta.' not in k and '.adapter_' not in k}
            torch.save(payload,path)
            teacher.load_checkpoint(path)
            validate_teacher_payload(teacher,payload)
            del payload['mot'][next(iter(payload['mot']))]
            with self.assertRaisesRegex(ValueError,'incomplete'):
                validate_teacher_payload(teacher,payload)

    def test_presence_and_visual_padding(self):
        for cm in (False,True):
            model=tiny_model(('action','video','dino','pointmap'))
            for name in ('video','dino','pointmap'):
                setattr(model.flex_joint,'p_present_'+name,0.)
                setattr(model.flex_joint,'cross_modal_predict_'+name,cm)
            data=batch()
            data['image_is_pad']=torch.ones(2,5,dtype=torch.bool)
            loss,metrics=model.training_loss(data)
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            for key,value in metrics.items():
                if key.startswith(('loss_flowmap_', 'loss_diagonal_')) and not key.endswith('action'):
                    self.assertEqual(value,0.)

    def test_random_initialization_resets_denoisers(self):
        model=tiny_model(('action','video'))
        norm=model.action_expert.blocks[0].self_attn.norm_q
        with torch.no_grad():
            norm.weight.fill_(7.)
        before=model.action_expert.head.weight.detach().clone()
        model.flow_map.initialization='random'
        model._flowmap_adaptation_installed=False
        install_adaptation(model)
        torch.testing.assert_close(norm.weight,torch.ones_like(norm.weight))
        self.assertFalse(torch.equal(before,model.action_expert.head.weight))
        self.assertEqual(float(model.action_expert.time_embedding_delta[-1].weight.norm()),0.)

    def test_tiny_joint_self_distillation_optimizes(self):
        torch.manual_seed(7)
        model=tiny_model(('action','video'))
        data=batch()
        optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
        losses=[]
        raw_losses=[]
        for _ in range(15):
            torch.manual_seed(321)
            optimizer.zero_grad()
            loss,metrics=model.training_loss(data)
            raw_losses.append(metrics["loss_unweighted"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            losses.append(float(loss.detach()))
        self.assertLess(losses[-1],losses[0])
        self.assertLess(raw_losses[-1],raw_losses[0])


if __name__=='__main__':unittest.main(verbosity=2)

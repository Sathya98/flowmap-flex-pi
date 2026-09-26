"""Controlled numerical tests for sampling, endpoint conversion and PFMM targets."""
import unittest
from unittest.mock import patch
import torch
from test_flowmap import tiny_model, batch
from flexpi.models.helpers.adaptation import clone_teacher
from flexpi.models.helpers.dino import _dino_x0_to_velocity
from flexpi.models.helpers.flowmap import FlowMapConfig, map_residuals, sample_level_pair_strip
from flexpi.models.helpers.flowmap_training import training_loss
from flexpi.models.helpers.flowmap_diagnostics import paired_time_proposals, make_noise, pfmm_probe, lmd_probe


class DiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(103)

    def test_sampling_distribution_and_low_sigma_mass(self):
        pairs = paired_time_proposals(200000)
        for kind, (s,t) in pairs.items():
            self.assertTrue(bool(((0 <= t) & (t <= s) & (s <= 1)).all()))
            expected = .05 if kind == 'conditional' else .05**2
            self.assertAlmostEqual(float((s < .05).float().mean()), expected, delta=.002)
            self.assertAlmostEqual(float(s.mean()), .5 if kind == 'conditional' else 2/3, delta=.003)

    def test_production_uniform_area_sampler(self):
        for lo, hi, width in ((0.,1.,1.), (0.,1.,.3), (0.,1.,.03), (.2,.8,.2), (.2,.8,1.)):
            with self.subTest(bounds=(lo,hi),width=width):
                s,t=sample_level_pair_strip(200000,width,'cpu',torch.bfloat16,
                    sigma_min=lo,sigma_max=hi,generator=torch.Generator().manual_seed(73))
                self.assertEqual(s.dtype,torch.float32)
                self.assertTrue(bool(((t>=lo-1e-7)&(t<=s)&(s<=hi)&(s-t<=width+1e-6)).all()))
                length=hi-lo;w=min(width,length)
                area=w*length-w*w/2
                expected_source=lo+(w*length*length/2-w**3/6)/area
                self.assertAlmostEqual(float(s.mean()),expected_source,delta=.002)
                self.assertAlmostEqual(float(t.mean()),lo+hi-expected_source,delta=.002)
                if (lo,hi,width)==(0.,1.,1.):
                    self.assertAlmostEqual(float((s<.05).float().mean()),.0025,delta=.0005)
                    self.assertAlmostEqual(float(((s<.75)&(t>.25)).float().mean()),.25,delta=.003)
                    g=torch.Generator().manual_seed(29)
                    a=torch.rand(200000,generator=g);b=torch.rand(200000,generator=g)
                    # Independent reference: Boffi's sorted-uniform construction.
                    for actual,reference in ((s,torch.maximum(a,b)),(t,torch.minimum(a,b))):
                        torch.testing.assert_close(torch.quantile(actual,torch.tensor([.1,.5,.9])),
                            torch.quantile(reference,torch.tensor([.1,.5,.9])),atol=.004,rtol=0)
        s,_=sample_level_pair_strip(200000,1.,'cpu',torch.float32,
            sampling='conditional',generator=torch.Generator().manual_seed(73))
        self.assertAlmostEqual(float(s.mean()),.5,delta=.002)
        self.assertAlmostEqual(float((s<.05).float().mean()),.05,delta=.002)

    def test_training_uses_configured_time_proposal(self):
        model=tiny_model(objective='lsd')
        for sampling in ('uniform_triangle','conditional'):
            model.flow_map.time_sampling=sampling
            with patch('flowmap_core.flowmap.sample_level_pair_strip',  # called via training_time_pairs
                       wraps=sample_level_pair_strip) as draw:
                data = batch(b=1)
                data["_flowmap_diagonal_mask"] = torch.zeros(1, dtype=torch.bool)
                loss,_=training_loss(model,data)
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertEqual(draw.call_args.kwargs['sampling'],sampling)
        with self.assertRaisesRegex(ValueError,'time_sampling'):
            FlowMapConfig(time_sampling='invalid')
        with self.assertRaisesRegex(ValueError,'time sampling'):
            sample_level_pair_strip(1,1.,'cpu',torch.float32,sampling='invalid')

    def test_endpoint_error_amplification_and_floor(self):
        s=torch.tensor([.5,.05,.005]).view(3,1)
        actual=_dino_x0_to_velocity(torch.zeros_like(s),torch.ones_like(s),s*1000,1000)
        torch.testing.assert_close(actual.square().flatten(),torch.tensor([4.,400.,400.]))

    def test_teacher_euler_convergence_on_known_ode(self):
        # x'=x, integrated backwards in sigma; exact endpoint exp(t-s)*x.
        x=(torch.ones(1,1,dtype=torch.float64),)
        s=torch.tensor([.9],dtype=torch.float64);t=torch.tensor([.1],dtype=torch.float64)
        errors=[]
        for steps in (8,16,32,64):
            trace={}
            map_residuals(lambda state,lo,hi:(torch.zeros_like(state[0]),),
                lambda state,time:state,x,s,t,FlowMapConfig(objective='pfmm',teacher_steps=steps),trace)
            endpoint=x[0]+(t-s)*trace['target_velocity'][0]
            errors.append(float((endpoint-torch.exp(t-s)*x[0]).abs().max()))
        self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))
        self.assertLess(errors[-1],.003)

    def test_fixed_inputs_reproduce_every_objective_loss_and_gradients(self):
        for objective in ('lmd','emd','pfmm','lsd','esd','psd_m','psd_u'):
            with self.subTest(objective=objective):
                model=tiny_model(streams=('action','video','dino','pointmap'),objective=objective)
                model.flow_map.teacher_steps=2
                if model.flow_map.needs_teacher:
                    object.__setattr__(model,'flow_map_teacher',clone_teacher(model))
                inputs=batch(b=1)
                noise=make_noise(inputs,123)
                results=[]
                for _ in range(2):
                    model.zero_grad(set_to_none=True)
                    torch.manual_seed(999) # also freeze PSD-U's intermediate time
                    loss,_=training_loss(model,inputs,times=(torch.tensor([.6]),torch.tensor([.2])),noise=noise)
                    loss.backward()
                    grads=torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                    self.assertTrue(bool(torch.isfinite(grads).all()))
                    results.append((loss.detach(),grads.clone()))
                for a,b in zip(*results):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_production_probe_matches_loss_and_endpoint_identity(self):
        model=tiny_model(streams=('action','video','dino','pointmap'),objective='pfmm')
        object.__setattr__(model,'flow_map_teacher',clone_teacher(model))
        model.eval()
        inputs=batch(b=1)
        report,targets=pfmm_probe(model,inputs,inputs,.5,.1,teacher_steps=2)
        self.assertTrue(report['finite'])
        self.assertEqual(set(targets),{'action','video','dino','pointmap'})
        d=report['dino_parameterization']
        self.assertAlmostEqual(d['fp32_conversion_residual_mse'],
            d['head_vs_implied_teacher_endpoint_mse']*4,delta=1e-4)
        for stream in report['streams'].values():
            self.assertAlmostEqual(stream['map_endpoint_residual_mse'],
                .16*stream['velocity_residual_mse'],delta=1e-4)
        second,_=pfmm_probe(model,inputs,inputs,.5,.1,teacher_steps=2)
        self.assertEqual(report,second)
        self.assertEqual(model.flow_map.teacher_steps,4)

    def test_lmd_probe_matches_objective_and_restores_pfmm(self):
        model=tiny_model(streams=('action','video','dino','pointmap'),objective='pfmm')
        teacher=clone_teacher(model)
        object.__setattr__(model,'flow_map_teacher',teacher)
        model.eval()
        inputs=batch(b=1)
        original=model.flow_map
        first=lmd_probe(model,inputs,inputs,.5,.1)
        self.assertTrue(first['finite'])
        self.assertIs(model.flow_map,original)
        self.assertIs(model.flow_map_teacher,teacher)
        self.assertEqual(first,lmd_probe(model,inputs,inputs,.5,.1))
        model.flow_map.objective='lmd'
        with torch.no_grad():
            loss,metrics=training_loss(model,inputs,prepared_inputs=inputs,
                times=(torch.tensor([.5]),torch.tensor([.1])),noise=make_noise(inputs,2026))
        self.assertEqual(first['loss'],float(loss))
        self.assertEqual(first['metrics'],metrics)
        model.flow_map.objective='pfmm'
        with self.assertRaises(ValueError):
            lmd_probe(model,inputs,inputs,.1,.5)
        self.assertIs(model.flow_map,original)
        self.assertEqual(model.flow_map.objective,'pfmm')


if __name__=='__main__':unittest.main()

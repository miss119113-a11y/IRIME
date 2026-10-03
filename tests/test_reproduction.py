"""Mathematical contracts and frozen-package regression checks (standard library runner)."""
import json
from pathlib import Path
import unittest
import numpy as np
from irime import irime, steering, _steering_derivatives, _orthogonal_complement
from experiments import paper_protocol as P

class ReproductionContracts(unittest.TestCase):
    def test_conditional_projector_identity(self):
        rng=np.random.default_rng(17)
        pos=P.circular_array_positions();points=np.array([[23.,32.],[141.,47.],[262.,61.]])
        a=steering(points,pos,1.0);q,_=_orthogonal_complement(a[:,:2],12)
        x=rng.standard_normal((12,40))+1j*rng.standard_normal((12,40));r=x@x.conj().T/40
        full,_=_orthogonal_complement(a,12)
        b=q@a[:,2];gain=np.vdot(b,r@b).real/np.vdot(b,b).real
        self.assertAlmostEqual(np.trace(full@r).real,np.trace(q@r).real-gain,places=10)

    def test_analytic_degree_derivatives(self):
        pos=P.circular_array_positions();c=np.array([147.2,52.6]);delta=1e-4
        _,dt,dp=_steering_derivatives(c,pos,1.0)
        for j,target in enumerate((dt,dp)):
            e=np.eye(2)[j]*delta
            finite=(steering(c+e,pos,1.0)[:,0]-steering(c-e,pos,1.0)[:,0])/(2*delta)
            np.testing.assert_allclose(target,finite,rtol=1e-7,atol=1e-9)

    def test_phase_aligned_symmetric_axis(self):
        z=np.array([1.2+.3j,.6-.4j])
        def axis(v):return np.real(v*np.exp(-.5j*np.angle(v.T@v)))
        u=axis(z)
        for phase in np.linspace(-np.pi,np.pi,11):
            v=axis(z*np.exp(1j*phase))
            self.assertLess(min(np.linalg.norm(v-u),np.linalg.norm(v+u)),1e-12)

    def test_exact_spherical_pair_spacing(self):
        for d in P.MAIN_SPACINGS:
            pair=P.fixed_elevation_azimuth_pair('main_mid',d)
            self.assertAlmostEqual(float(P.great_circle(pair[:1],pair[1:])[0,0]),d,places=9)

    def test_missing_directions_remain_in_error_metrics(self):
        cfg=P.paper_case();a=P.pair_doa_errors(cfg['theta_true'],cfg['phi_true'],[],[])
        b=P.spherical_match(cfg['theta_true'],cfg['phi_true'],[],[])
        self.assertAlmostEqual(a['joint_rmse'],np.sqrt(180**2+90**2))
        self.assertEqual(b['trial_rmse_deg'],180.0)
        self.assertEqual(b['complete_3deg'],0.0)

    def test_frozen_package_estimates(self):
        fixtures=json.loads((Path(__file__).parent/'frozen_fixtures.json').read_text())
        for case in fixtures:
            family=case['family'];v=case['value'];t=case['local_trial']
            if family in ('snr','snapshots'):data=P.main_observation(family,v,t)
            elif family=='sources':data=P.source_observation(v,t)
            else:data=P.resolution_observation(v[0],v[1],t)
            cfg,pos,_,x,seed,_,_=data
            theta,phi=irime(x,pos,1.0,cfg['theta_range'],cfg['phi_range'],cfg['source_num'],seed=seed)
            np.testing.assert_allclose(theta,case['theta'],rtol=0,atol=2e-5)
            np.testing.assert_allclose(phi,case['phi'],rtol=0,atol=2e-5)

if __name__=='__main__':unittest.main()

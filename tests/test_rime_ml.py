import unittest
import numpy as np
from experiments.rime_ml import RawDML, rime_ml, _propose
from experiments import paper_protocol as P


class RimeMLTests(unittest.TestCase):
    def test_raw_projector_matches_pseudoinverse_including_duplicate_sources(self):
        cfg,pos,s,x,seed,sigma,trial=P.source_observation(9,0)
        objective=RawDML(x,pos,1,9)
        rng=np.random.default_rng(31)
        vectors=rng.uniform(0,1,(8,18))*np.r_[np.full(9,360),np.full(9,90)]
        vectors[1,1],vectors[1,10]=vectors[1,0],vectors[1,9]
        losses=objective(vectors)
        for a,loss in zip(objective.steering_batch(vectors),losses):
            projector=np.eye(len(pos))-a@np.linalg.pinv(a,rcond=1e-10)
            expected=np.trace(projector@objective.covariance).real/objective.trace
            self.assertAlmostEqual(loss,expected,places=11)
        self.assertEqual(objective.evaluations,8)

    def test_permutation_and_azimuth_turn_preserve_raw_loss(self):
        cfg,pos,s,x,seed,sigma,trial=P.source_observation(3,2)
        objective=RawDML(x,pos,1,3)
        v=np.r_[cfg['theta_true'],cfg['phi_true']]
        swapped=np.r_[v[:3][[2,0,1]]+360,v[3:][[2,0,1]]]
        np.testing.assert_allclose(objective([v,swapped]),[objective(v)[0]]*2,atol=2e-15)

    def test_official_proposal_matches_scalar_equations_and_matlab_round(self):
        class Tape:
            def __init__(self):self.calls=0
            def random(self,shape=None):
                self.calls+=1
                if shape is None:return .72
                if self.calls==2:return np.full(shape,.2)
                if self.calls==3:return np.arange(np.prod(shape)).reshape(shape)/100+.01
                return np.array([[.5,.2,.5,.2],[.97,.5,.97,.5]])
        points=np.array([[100.,20.,80.,50.],[200.,10.,70.,45.]])
        best=points[0].copy();fit=np.array([.2,.5]);lb=np.zeros(4);ub=np.array([360,360,90,90])
        tape=Tape();candidate=_propose(points,best,fit,lb,ub,1,10,tape)
        positions_random=np.arange(8).reshape(2,4)/100+.01
        hard_random=np.array([[.5,.2,.5,.2],[.97,.5,.97,.5]])
        # 1*5/10 = .5 must round to 1, whereas Python round would give 0.
        factor=(.72-.5)*2*np.cos(np.pi)*(1-1/5)
        expected=points.copy()
        for i in range(2):
            for j in range(4):
                if .2<np.sqrt(.1):expected[i,j]=best[j]+factor*ub[j]*positions_random[i,j]
                if hard_random[i,j]<fit[i]/np.linalg.norm(fit):expected[i,j]=best[j]
        np.testing.assert_allclose(candidate,np.clip(expected,lb,ub),atol=1e-13)

    def test_reproducible_count_and_greedy_acceptance(self):
        cfg,pos,s,x,seed,sigma,trial=P.main_observation('snr',-5,0)
        a=rime_ml(x,pos,1,3,seed=99,population=7,iterations=13,return_diagnostics=True)
        b=rime_ml(x,pos,1,3,seed=99,population=7,iterations=13,return_diagnostics=True)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])
        self.assertEqual(a[2],b[2]);self.assertEqual(a[2]['objective_evaluations'],98)
        self.assertLessEqual(a[2]['final_loss'],a[2]['initial_loss'])
        check=RawDML(x,pos,1,3)(np.r_[a[0],a[1]])[0]
        self.assertAlmostEqual(check,a[2]['final_loss'],places=14)

if __name__=='__main__':unittest.main()

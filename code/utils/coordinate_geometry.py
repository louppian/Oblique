"""Mean/noise coordinates and the orthogonal complement of their plane."""
import numpy as np
from geometry import unit, basis, residual

class CoordinateAxes:
    def __init__(self,mean,noise):
        self.r_I=np.asarray(mean,dtype=float);self.R_I=float(np.linalg.norm(mean))
        self.r_hat=unit(mean);self.b=unit(noise);self.gamma=float(self.r_hat@self.b)
        self.q=basis([self.r_hat,self.b]);self.length=float(np.linalg.norm(self.r_hat-self.b))
        if self.q.shape[1]!=2:raise ValueError('Mean/noise plane requires two independent directions')
        self.m_hat=(self.r_hat-self.b)/self.length
    def coords(self,z):
        z=unit(z);c=z@self.r_hat;n=z@self.b
        return dict(c=c,n=n,h=(c-n+1-self.gamma)/(2*(1-self.gamma)),resid=np.sqrt(np.maximum(0,1-np.sum((z@self.q)**2,axis=1))))

def geometry_stats(reference,noise,evaluation):
    a=CoordinateAxes(reference.mean(0),noise.mean(0));s=a.coords(evaluation)
    angles=np.degrees(np.arccos(np.clip(s['n'],-1,1)))
    return dict(R_b=float(np.linalg.norm(noise.mean(0))),R_I=a.R_I,gamma=a.gamma,
        angle_b_r_deg=float(np.degrees(np.arccos(np.clip(a.gamma,-1,1)))),
        phi_deg=float(np.degrees(np.arccos(np.clip(a.r_hat@a.m_hat,-1,1)))),
        gamma_percentile=float(100*np.mean(s['c']<a.gamma)),
        frac_noise_c_above_q01=float(np.mean(noise@a.r_hat>np.quantile(s['c'],0.01))),
        nearest_data_angle_deg=float(angles.min()),data_angle_q01_deg=float(np.quantile(angles,0.01)),
        plane_residual_norm_mean=float(s['resid'].mean()))

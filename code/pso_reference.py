#!/usr/bin/env python3
"""NEW Python interpretation of revised Step3 *group-level* PSO, not MATLAB parity.

Feature definition and group hierarchy were read from archived Step3 MATLAB code.
Random generator is NumPy PCG64, NOT MATLAB twister. Values/trajectories are not
expected to match MATLAB under an identical numeric seed. Explicit SK04 key
stores all selected groups, allowing reconstruction without PSO during extraction.
"""
from dataclasses import dataclass
import math
import numpy as np
from scipy.ndimage import convolve
from reference_codec import canonical_image

@dataclass(frozen=True)
class Config:
    alpha: float=1/3
    beta: float=1/3
    gamma: float=1/3
    swarm_size: int=30
    max_iter: int=100
    w_start: float=0.9
    w_end: float=0.4
    c1: float=2.0
    c2: float=2.0
    seed: int=20260918

def normalize01(a):
    a=np.asarray(a,dtype=np.float64)
    lo,hi=float(a.min()),float(a.max())
    if not np.isfinite([lo,hi]).all(): raise ValueError('Nonfinite input')
    return (a-lo)/(hi-lo) if hi>lo else np.zeros_like(a)

def fitness_map(plane, cfg):
    X=normalize01(plane)
    K=np.ones((3,3),dtype=np.float64)/9.
    # MATLAB conv2(...,'same') corresponds to convolution w/ zero boundary.
    mu=convolve(X,K,mode='constant',cval=0.)
    mu2=convolve(X*X,K,mode='constant',cval=0.)
    var=np.maximum(0,mu2-mu*mu)
    sx=np.array([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=np.float64)/8.
    gx=convolve(X,sx,mode='constant',cval=0.)
    gy=convolve(X,sx.T,mode='constant',cval=0.)
    grad=np.sqrt(gx*gx+gy*gy)
    edge=convolve(grad,K,mode='constant',cval=0.)
    contrast=convolve(np.abs(X-mu),K,mode='constant',cval=0.)
    return cfg.alpha*normalize01(var)+cfg.beta*normalize01(edge)+cfg.gamma*normalize01(contrast)

def group_fitness(cover,cfg):
    cover=canonical_image(cover)
    if min(cfg.alpha,cfg.beta,cfg.gamma)<0 or not math.isclose(cfg.alpha+cfg.beta+cfg.gamma,1,abs_tol=1e-10):
        raise ValueError('Invalid fitness weights')
    channels=1 if cover.ndim==2 else 3
    out=[]
    for ch in range(channels):
        plane=cover if channels==1 else cover[:,:,ch]
        score=fitness_map(plane,cfg)
        block_mean=score.reshape(64,8,64,8).mean(axis=(1,3))
        out.append(block_mean.reshape(-1))
    fitness=np.concatenate(out)
    if not np.isfinite(fitness).all() or len(fitness)!=4096*channels:
        raise AssertionError('Fitness invalid')
    return fitness

def select_groups(fitness,needed,channels,cfg):
    fitness=np.asarray(fitness,dtype=np.float64)
    if len(fitness)!=4096*channels or not np.isfinite(fitness).all():
        raise ValueError('Unexpected group fitness array')
    if not isinstance(needed,int) or not 0<=needed<=len(fitness):
        raise ValueError('Requested group count invalid')
    if cfg.swarm_size<1 or cfg.max_iter<1: raise ValueError('PSO shape invalid')
    rng=np.random.default_rng(cfg.seed)
    available=np.ones(len(fitness),dtype=bool)
    selected=[]; trace=[]
    S,T=cfg.swarm_size,cfg.max_iter
    for s in range(needed):
        dims=np.array([64.,64.,float(channels)])
        pos=1+(dims-1)*rng.random((S,3))
        vel=np.zeros_like(pos)
        pbest=pos.copy()
        pfit=np.full(S,-np.inf)
        gbest=pos[0].copy()
        gfit=-np.inf; gid=-1
        for t in range(T):
            discrete=np.floor(pos+0.5).astype(np.int64)
            discrete=np.maximum(1,np.minimum(discrete,dims.astype(np.int64)))
            ids=(discrete[:,2]-1)*4096+(discrete[:,0]-1)*64+(discrete[:,1]-1)
            fit=np.where(available[ids],fitness[ids],-np.inf)
            improve=fit>pfit
            pbest[improve]=pos[improve];pfit[improve]=fit[improve]
            j=int(np.argmax(fit))
            if fit[j]>gfit:
                gfit=float(fit[j]);gbest=pos[j].copy();gid=int(ids[j])
            w=cfg.w_end if T==1 else cfg.w_start-(cfg.w_start-cfg.w_end)*t/(T-1)
            r1=rng.random((S,3));r2=rng.random((S,3))
            vel=w*vel + cfg.c1*r1*(pbest-pos)+cfg.c2*r2*(gbest-pos)
            pos=np.clip(pos+vel,1.,dims)
        fallback=False
        if gid<0 or not available[gid] or not np.isfinite(gfit):
            candidates=np.flatnonzero(available)
            gid=int(candidates[np.argmax(fitness[candidates])])
            gfit=float(fitness[gid]);fallback=True
        available[gid]=False
        selected.append(gid)
        ch=gid//4096+1
        br=(gid%4096)//64;bc=gid%64
        rnd=(br//8)*8+(bc//8)+1
        brnd=(br%8)*8+(bc%8)+1
        trace.append(dict(selection_order=s+1,group_id=gid+1,block_row8=br+1,
            block_col8=bc+1,channel=ch,RND=rnd,BRND=brnd,
            fitness=gfit,fallback_used=fallback))
    return selected,trace

def pso_groups(cover,nbits,cfg=Config()):
    cover=canonical_image(cover)
    if nbits<0 or nbits>cover.size or nbits!=int(nbits): raise ValueError('Capacity overflow')
    count=math.ceil(nbits/64)
    ch=1 if cover.ndim==2 else 3
    scores=group_fitness(cover,cfg)
    ids,trace=select_groups(scores,count,ch,cfg)
    return [(d['RND'],d['BRND'],d['channel']) for d in trace],trace,scores

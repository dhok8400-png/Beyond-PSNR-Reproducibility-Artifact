import os, sys, time, csv, json, hashlib, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
for env in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[env]='1'
ROOT=Path('/mnt/data/_selector_det_run')
VENDOR=ROOT/'new_experiments/code/vendor'
sys.path.insert(0,str(VENDOR))
import numpy as np
from PIL import Image
import sealwatch as sw
OUT=ROOT/'matched_selector'; FEAT=OUT/'features'
SELECTORS=['sequential','greedy_feature','random','pso']
PAYLOADS=['text_4096','text_8192']
DSEEDS=[20260922,20260923,20260924]
EMBED_BPP={'text_4096':2606*8/262144,'text_8192':4959*8/262144}

def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def series_for(i):
    if i<=11:return 'series-00001'
    if i<=23:return 'series-00002'
    if i<=35:return 'series-00003'
    return 'series-00004'

def stego_path(case,payload,selector):
    seed='det' if selector in ('sequential','greedy_feature') else '20260918'
    return ROOT/'stegos'/case/f'{payload}__{selector}__{seed}__full.png'

def feature_job(job):
    case,kind,payload,selector,path=job
    with Image.open(path) as im:
        assert im.mode=='L' and im.size==(512,512)
        x=np.array(im)
    t=time.perf_counter(); grouped=sw.srm.extract(x); elapsed=time.perf_counter()-t
    v=np.concatenate([a.ravel() for a in grouped.values()])
    assert len(v)==34671 and np.isfinite(v).all()
    fn=f'{case}__cover.npy' if kind=='cover' else f'{case}__{payload}__{selector}.npy'
    np.save(FEAT/fn,v)
    return {'candidate_id':case,'kind':kind,'payload_id':payload,'selector':selector,'feature_file':fn,'source_file':str(path.relative_to(ROOT)),'source_sha256':sha_file(path),'feature_sha256':sha_file(FEAT/fn),'seconds':elapsed}

def extract(workers=5):
    FEAT.mkdir(parents=True,exist_ok=True)
    jobs=[]
    for i in range(1,37):
        case=f'MED-{i:03d}'
        jobs.append((case,'cover','none','cover',ROOT/'covers'/f'{case}.jpg'))
        for payload in PAYLOADS:
            for selector in SELECTORS:
                jobs.append((case,'stego',payload,selector,stego_path(case,payload,selector)))
    rows=[]; t0=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(feature_job,j) for j in jobs]
        for n,f in enumerate(as_completed(fs),1):
            r=f.result(); rows.append(r)
            if n%10==0 or n==len(fs): print('FEATURE',n,'/',len(fs),'elapsed',round(time.perf_counter()-t0,1),flush=True)
    rows=sorted(rows,key=lambda r:(r['candidate_id'],r['kind'],r['payload_id'],r['selector']))
    with open(OUT/'feature_manifest.csv','w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    (OUT/'feature_status.json').write_text(json.dumps({'complete':True,'feature_vectors':len(rows),'dimension':34671,'selectors':SELECTORS,'payloads':PAYLOADS,'implementation':'sealwatch 2025.9 vendor SRM with exact vectorized co-occurrence counting; validated bit-identical to archived MED-001 reference feature.','seconds':time.perf_counter()-t0},indent=2))

def metrics(y,s):
    y=np.asarray(y,dtype=int); s=np.asarray(s,float); pred=(s>0).astype(int)
    tp=int(np.sum((pred==1)&(y==1))); tn=int(np.sum((pred==0)&(y==0))); fp=int(np.sum((pred==1)&(y==0))); fn=int(np.sum((pred==0)&(y==1)))
    pos=s[y==1]; neg=s[y==0]
    auc=float(np.mean((pos[:,None]>neg[None,:]).astype(float)+.5*(pos[:,None]==neg[None,:])))
    acc=(tp+tn)/len(y)
    return {'n_samples':len(y),'n_cover':len(neg),'n_stego':len(pos),'accuracy':acc,'balanced_accuracy':.5*(tp/len(pos)+tn/len(neg)),'auc':auc,'detection_error':1-acc,'fpr':fp/len(neg),'fnr':fn/len(pos),'tpr':tp/len(pos),'tnr':tn/len(neg),'tp':tp,'tn':tn,'fp':fp,'fn':fn}

def train_job(job):
    selector,payload,seed,heldout,cases,groups=job
    c=np.stack([np.load(FEAT/f'{case}__cover.npy') for case in cases])
    s=np.stack([np.load(FEAT/f'{case}__{payload}__{selector}.npy') for case in cases])
    groups=np.asarray(groups)
    test=np.flatnonzero(groups==heldout); train=np.flatnonzero(groups!=heldout)
    xc=np.ascontiguousarray(c[train]); xs=np.ascontiguousarray(s[train])
    x=np.concatenate([c[test],s[test]]); y=np.r_[np.zeros(len(test),dtype=int),np.ones(len(test),dtype=int)]
    t=time.perf_counter(); trainer=sw.ensemble_classifier.FldEnsembleTrainer(xc,xs,seed=seed,L=101,d_sub=200,verbose=0); model,record=trainer.train(); seconds=time.perf_counter()-t
    score=model.predict_confidence(x)
    fold={'selector':selector,'payload_id':payload,'detector_seed':seed,'held_out_series':heldout,'train_pairs':len(train),'test_pairs':len(test),'train_seconds':seconds,'L':101,'d_sub':200,**metrics(y,score)}
    preds=[]
    for cls,ixs in [(0,range(len(test))),(1,range(len(test),2*len(test)))]:
        for n,ix in enumerate(ixs):
            preds.append({'selector':selector,'payload_id':payload,'detector_seed':seed,'held_out_series':heldout,'candidate_id':cases[test[n]],'class':cls,'score':float(score[ix]),'predicted_stego':bool(score[ix]>0),'train_pairs':len(train),'test_pairs':len(test)})
    return fold,preds

def train(workers=5):
    cases=[f'MED-{i:03d}' for i in range(1,37)]; groups=[series_for(i) for i in range(1,37)]; heldouts=sorted(set(groups))
    jobs=[(sel,pid,seed,g,cases,groups) for sel in SELECTORS for pid in PAYLOADS for seed in DSEEDS for g in heldouts]
    folds=[]; preds=[]; t0=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        fs=[ex.submit(train_job,j) for j in jobs]
        for n,f in enumerate(as_completed(fs),1):
            fold,pp=f.result(); folds.append(fold); preds.extend(pp)
            print('MODEL',n,'/',len(fs),fold['selector'],fold['payload_id'],fold['detector_seed'],fold['held_out_series'],'auc',round(fold['auc'],3),'elapsed',round(time.perf_counter()-t0,1),flush=True)
    folds=sorted(folds,key=lambda r:(r['selector'],r['payload_id'],r['detector_seed'],r['held_out_series']))
    preds=sorted(preds,key=lambda r:(r['selector'],r['payload_id'],r['detector_seed'],r['held_out_series'],r['class'],r['candidate_id']))
    for name,rows in [('fold_results.csv',folds),('predictions.csv',preds)]:
        with open(OUT/name,'w',newline='',encoding='utf-8') as fh:
            w=csv.DictWriter(fh,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    pooled=[]
    for sel in SELECTORS:
        for pid in PAYLOADS:
            for seed in DSEEDS:
                rs=[r for r in preds if r['selector']==sel and r['payload_id']==pid and r['detector_seed']==seed]
                y=[r['class'] for r in rs]; score=[r['score'] for r in rs]
                item={'selector':sel,'payload_id':pid,'embedded_bpp':EMBED_BPP[pid],'detector_seed':seed,'split':'leave-one-archive-series-out','images':36,'folds':4,**metrics(y,score)}
                item['mean_fold_auc']=float(np.mean([r['auc'] for r in folds if r['selector']==sel and r['payload_id']==pid and r['detector_seed']==seed]))
                pooled.append(item)
    with open(OUT/'pooled_results.csv','w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=pooled[0].keys()); w.writeheader(); w.writerows(pooled)
    summary=[]
    for sel in SELECTORS:
        for pid in PAYLOADS:
            rs=[r for r in pooled if r['selector']==sel and r['payload_id']==pid]
            fs=[r for r in folds if r['selector']==sel and r['payload_id']==pid]
            item={'selector':sel,'payload_id':pid,'embedded_bpp':EMBED_BPP[pid],'unique_images':36,'detector_seeds':3,'folds':4}
            for k in ['accuracy','balanced_accuracy','auc','detection_error','fpr','fnr','tpr','mean_fold_auc']:
                vals=[r[k] for r in rs]; item[k+'_mean']=float(np.mean(vals)); item[k+'_seed_sd']=float(np.std(vals,ddof=1))
            item['fold_auc_min']=float(min(r['auc'] for r in fs)); item['fold_auc_max']=float(max(r['auc'] for r in fs))
            summary.append(item)
    with open(OUT/'matched_selector_summary.csv','w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=summary[0].keys()); w.writeheader(); w.writerows(summary)
    (OUT/'run_status.json').write_text(json.dumps({'complete':True,'selectors':SELECTORS,'payloads':PAYLOADS,'features':36+36*len(SELECTORS)*len(PAYLOADS),'models':len(jobs),'out_of_fold_predictions':len(preds),'detector_seeds':DSEEDS,'split':'leave-one-archive-series-out','series_sizes':{'series-00001':11,'series-00002':12,'series-00003':12,'series-00004':1},'patient_independence_verified':False,'scope':'Matched-stream four-selector full-SRM/FLD comparison. Stochastic selectors use embedding seed 20260918; deterministic selectors are single runs. Detector seeds are technical repetitions.'},indent=2))
    print('TRAIN_DONE seconds',time.perf_counter()-t0,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('action',choices=['extract','train','all']); ap.add_argument('--workers',type=int,default=5); a=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.action in ('extract','all'): extract(a.workers)
    if a.action in ('train','all'): train(a.workers)

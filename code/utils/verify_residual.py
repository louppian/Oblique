"""CPU verification of the definitions and all four experiment entry points."""
import csv
import importlib.util
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np


def main():
    path=Path(__file__).with_name('residual_core.py')
    spec=importlib.util.spec_from_file_location('new_residual',path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    rng=np.random.default_rng(42)
    d=12
    def clean(n):
        x=rng.normal(size=(n,d))*np.linspace(.6,.1,d)
        x[:,0]+=2
        x[:,3]*=3
        return m.unit(x)
    train=clean(120)
    text=m.unit(train+.03*rng.normal(size=train.shape))
    noise=m.unit(rng.normal(size=(50,d))*.02+np.eye(d)[1])
    nt=m.unit(noise+.01*rng.normal(size=noise.shape))
    frame=m.fit_frame(train,text,noise,nt)
    for modality,x in [('image',train),('text',text)]:
        f=frame[modality]
        parts=m.components(x,f)
        np.testing.assert_allclose(sum(parts.values()),x,atol=1e-12)
        np.testing.assert_allclose(parts['Delta'] @ f['q3'],0,atol=1e-12)
        np.testing.assert_allclose(f['q'].T @ f['e'],0,atol=1e-12)
        cov=m.covariance(m.residual(x,f['q']))
        np.testing.assert_allclose(cov @ f['e'],f['e_eigenvalue']*f['e'],atol=1e-12)
        coefficients=x @ f['dual']
        np.testing.assert_allclose(coefficients @ np.stack((f['r'],f['b'])),
            (x @ f['q']) @ f['q'].T,atol=1e-12)
        # An individual outside-plane vector need not be parallel to the shared PCA axis.
        assert np.mean(np.linalg.norm(parts['Delta'],axis=1))>1e-3
    # rhat and bhat are oblique: dot products alone do not give their coefficients.
    f=frame['image']
    u=1.3*f['r']-.4*f['b']+.8*f['e']
    np.testing.assert_allclose(u @ f['dual'],[1.3,-.4],atol=1e-12)
    np.testing.assert_allclose(m.components(u,f)['Delta'],0,atol=1e-12)
    # Independent definitions of both sides verify component scores, including cross terms.
    z=clean(8)
    mapping=np.repeat(np.arange(8),2)
    t=m.unit(z[mapping]+rng.normal(size=(16,d))*.1)
    rows=np.arange(8)
    terms=m.pair_terms(z,t,rows,rows*2,frame)
    np.testing.assert_allclose(sum(terms[k] for k in ('Rhat','Bhat','E','Delta')),
        np.einsum('ij,ij->i',z,t[::2]),atol=1e-12)
    cfg=replace(m.CONFIG,epochs=1,seeds=(0,),reference_modes=('initial_fixed','epoch_current'),
        random_subspace_repeats=2,pair_shuffle_repeats=3,query_detail_epochs=(0,1),neighbor_k=3)
    meta=dict(run_id='clip_only_0.000_seed_0',seed=0,method='clip_only',lambda_axis=0.,epoch=0)
    initial=m.Snapshot(meta,z,t,z @ t.T,mapping,np.arange(8),np.arange(16),frame)
    with tempfile.TemporaryDirectory(prefix='residual_verify_',dir=path.parent) as tmp:
        context=m.AnalysisContext(cfg,tmp,initial)
        try:
            for method,epoch in [('clip_only',0),('clip_only',1),('conformity',0),('conformity',1)]:
                zz=m.unit(z+epoch*rng.normal(size=z.shape)*.04)
                tt=m.unit(t+epoch*rng.normal(size=t.shape)*.04)
                metadata=dict(meta,method=method,epoch=epoch,lambda_axis=0. if method=='clip_only' else .1,
                    run_id=f'{method}_seed_0')
                current_frame=m.fit_frame(m.unit(train+epoch*rng.normal(size=train.shape)*.03),
                    m.unit(text+epoch*rng.normal(size=text.shape)*.03),noise,nt)
                current=m.Snapshot(metadata,zz,tt,zz @ tt.T,mapping,initial.image_ids,initial.caption_ids,current_frame)
                for experiment in m.EXPERIMENTS:
                    experiment(current,context,render=method=='conformity' and epoch==1)
            for name,table in context.tables.items():
                assert table.summaries,name
                for row in table.summaries:
                    for key in ('reconstruction_max_error','score_reconstruction_max_error',
                                'variance_reconstruction_error','embedding_reconstruction_max_error'):
                        if key in row: assert row[key]<1e-8,(name,key,row[key])
        finally:
            context.close()
        files=list(Path(tmp).iterdir())
        assert len(files)==8,[p.name for p in files]
        for file in files:
            assert file.stat().st_size>100,file
        with (Path(tmp)/'h1_exp2_margin_decomposition.csv').open(encoding='utf-8-sig') as stream:
            rows=list(csv.DictReader(stream))
        assert any(row['candidate_policy']=='clip_only_fixed_candidates' for row in rows)
        assert all(key in rows[0] for key in ('Rhat_margin','Bhat_margin','E_margin','Delta_margin'))
        print('PASS: oblique coefficients, outside-plane PCA, orthogonality, exact scores,')
        print('      four experiments, fixed/current references, candidate controls, CSV/PNG outputs.')


if __name__=='__main__':
    main()

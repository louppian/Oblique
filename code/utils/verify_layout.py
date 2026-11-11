"""Small numerical and output-contract checks; no model training or real data."""
from pathlib import Path
import importlib
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cached_analysis as analysis
import experiment_runner as runner
import residual_core as core
from dataclasses import replace


def main():
    rng=np.random.default_rng(7)
    unit=lambda x:x/np.linalg.norm(x,axis=1,keepdims=True)
    z=unit(rng.normal(size=(40,8))+np.eye(8)[0]*2)
    t=unit(np.repeat(z,2,axis=0)+rng.normal(size=(80,8))*.1)
    noise=dict(image_unit=unit(rng.normal(size=(30,8))*.2+np.eye(8)[1]),
               text_unit=unit(rng.normal(size=(30,8))*.2+np.eye(8)[2]))
    image=dict(unit=z,ids=np.arange(40).astype(str),labels=(rng.random((40,2))>.5).astype(int))
    coco=dict(unit=t,ids=np.arange(80).astype(str),image_id=np.repeat(image['ids'],2))
    nih=dict(unit=unit(rng.normal(size=(8,8))+np.eye(8)[0]),ids=np.arange(8).astype(str),
             texts=np.asarray([f'prompt {i}' for i in range(8)]),polarity=np.array(['pos']*4+['neg']*4),
             disease=np.tile([0,0,1,1],2),template=np.tile([0,1,0,1],2))
    old_input,old_root=analysis.inputs,runner.ROOT
    with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
        runner.ROOT=Path(tmp)
        analysis.inputs=lambda model,dataset:(image,coco if dataset=='coco' else nih,noise)
        try:
            analysis.run_magnitude('magnitude',(('clip','coco'),('carzero','nih')),(0,),12)
            analysis.run_medical('medical',('carzero',),(0,),12)
            cfg=replace(core.CONFIG,seeds=(0,),max_train_images=12,random_subspace_repeats=1,pair_shuffle_repeats=2,neighbor_k=3)
            analysis.run_reference_controls('controls',('clip',),cfg)
            # Use the real legacy output collector and then the common publisher.
            an=importlib.import_module('1_analysis')
            with runner.workspace('sub','legacy') as work:
                with an.ExperimentOutput(work/'legacy','one'):
                    an.save_csv(work,'summary',[dict(value=1)])
                    fig,ax=plt.subplots();ax.plot([0,1],[0,1]);an.save_figure(work,fig,'plot')
            for group,name in [('main','magnitude'),('main','medical'),('sub','controls'),('sub','legacy')]:
                paths=list((Path(tmp)/'results'/group).glob(name+'.*'))
                assert {p.suffix for p in paths}=={'.csv','.png'} and len(paths)==2
                frame=pd.read_csv(Path(tmp)/'results'/group/(name+'.csv'))
                assert len(frame)>0 and 'analysis' in frame
            # Known independent metric example: perfect separation.
            result=analysis.metrics([0,0,1,1],np.array([-2,-1,1,2]))
            assert result['auc']==result['f1']==result['mcc']==1
        finally:analysis.inputs,runner.ROOT=old_input,old_root
    print('PASS: new geometry, medical metrics, reference controls, legacy collector, one CSV/PNG per runner')


if __name__=='__main__':main()

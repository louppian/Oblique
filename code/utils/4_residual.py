"""섹션 4: 잔차와 검색. 네 실험 함수, 실험별 CSV/PNG 하나.
alignment.exp12_alignment_training()가 동일 에폭 임베딩을 네 함수에 전달한다.
python 4_residual.py 실행도 동일한 공유 학습을 시작한다. CLI/JSON 출력 없음.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import importlib
al = importlib.import_module("3_alignment")
Config = al.Config
Snapshot = al.Snapshot
AnalysisContext = al.AnalysisContext
EPS = al.EPS
COMMON_COLUMNS = al.COMMON_COLUMNS
RETRIEVAL_COLUMNS = al.RETRIEVAL_COLUMNS
GEOMETRY_COLUMNS = al.GEOMETRY_COLUMNS
unit = al.unit
array = al.array
basis = al.basis
covariance = al.covariance
query_stats = al.query_stats
retrieval_summary = al.retrieval_summary
weighted_pearson = al.weighted_pearson
residual = al.residual
nearest_neighbors = al.nearest_neighbors
neighbor_overlap = al.neighbor_overlap
text_per_image = al.text_per_image
spectral_stats = al.spectral_stats
geometry = al.geometry
make_figure = al.make_figure
save_figure = al.save_figure
label = al.label
plot_curves = al.plot_curves
def h1_exp1_subspace_retrieval(snapshot, context, render=False):
    """H1-1. 축/평면과 잔차의 raw dot/cosine 검색; rank-matched random/PCA 대조."""
    extra = ['subspace', 'control_trial', 'subspace_rank', 'score_variant',
             'zero_image_rows', 'zero_text_rows', 'valid_score', 'score_reconstruction_max_error']
    table = context.table('h1_exp1_subspace_retrieval', extra + RETRIEVAL_COLUMNS)
    for mode, frame in context.frames(snapshot):
        meta = context.meta(snapshot, mode)
        metrics = retrieval_summary(query_stats(snapshot.scores, snapshot.mapping))
        table.append(dict(**meta, record_type='summary', subspace='full', control_trial=-1,
                          subspace_rank=snapshot.z.shape[1], score_variant='original',
                          zero_image_rows=0, zero_text_rows=0, valid_score=True,
                          score_reconstruction_max_error=0.000, **metrics))
        spaces = [(name, -1, q) for name, q in frame['spaces'].items()]
        # 각 실제 부분공간의 rank에 맞춘 대조군. 2차원/4차원을 섞어 비교하지 않는다.
        for rank in sorted({q.shape[1] for q in frame['spaces'].values()}):
            spaces.append(('pca_rank_matched', -1, frame['pca'][:, :rank]))
            spaces.extend(('random_rank_matched', trial, context.random_basis(snapshot.z.shape[1], rank, trial))
                          for trial in range(context.cfg.random_subspace_repeats))
        for name, trial, q in spaces:
            zi, ti = snapshot.z @ q, snapshot.t @ q
            plane = snapshot.scores.copy() if q.shape[1]==snapshot.z.shape[1] else zi @ ti.T
            outside = snapshot.scores - plane
            nz = np.linalg.norm(residual(snapshot.z, q), axis=1)
            nt = np.linalg.norm(residual(snapshot.t, q), axis=1)
            for variant, score in [('plane_raw', plane), ('residual_raw', outside),
                                   ('residual_cosine', None)]:
                valid = variant != 'residual_cosine' or (np.all(nz > EPS) and np.all(nt > EPS))
                if variant == 'residual_cosine' and valid:
                    score = outside / (nz[:, None]*nt[None, :])
                row = dict(**meta, record_type='summary', subspace=name, control_trial=trial,
                           subspace_rank=q.shape[1], score_variant=variant,
                           zero_image_rows=int((nz <= EPS).sum()), zero_text_rows=int((nt <= EPS).sum()),
                           valid_score=bool(valid), score_reconstruction_max_error=float(np.max(np.abs(plane+outside-snapshot.scores))))
                if valid:
                    row.update(retrieval_summary(query_stats(score, snapshot.mapping)))
                table.append(row)
    table.flush()
    if render:
        fig, axes = make_figure()
        for ax, direction in zip(axes[0], ('i2t','t2i')):
            joint_ranks={(r['run_id'],r['epoch']):r['subspace_rank'] for r in table.summaries
                         if r['reference_mode']=='epoch_current' and r['subspace']=='joint_cone'}
            final = [r for r in table.summaries if r['epoch'] == context.cfg.epochs and
                     r['reference_mode'] == 'epoch_current' and r['score_variant'] in ('original','residual_cosine') and
                     r['subspace'] in ('full','joint_cone','random_rank_matched','pca_rank_matched') and r['valid_score'] and
                     (r['subspace'] in ('full','joint_cone') or r['subspace_rank']==joint_ranks[(r['run_id'],r['epoch'])])]
            names = list(dict.fromkeys(label(r) for r in final))
            variants = ['full','joint_cone','random_rank_matched','pca_rank_matched']
            width = 0.8/max(len(names),1)
            for j, run in enumerate(names):
                means = [np.mean([r[f'{direction}_R1_percent'] for r in final if label(r)==run and r['subspace']==v]) for v in variants]
                ax.bar(np.arange(4)-0.4+width/2+j*width, means, width, label=run)
            ax.set_xticks(range(4), ['Original','Cone residual','Random residual','PCA residual'], rotation=15)
            ax.set(title=f'{direction.upper()}: final R@1', ylabel='Recall (%)')
            if names: ax.legend(fontsize=7)
        predicate = lambda r: r['reference_mode']=='epoch_current' and r['subspace']=='joint_cone' and r['score_variant']=='residual_raw'
        plot_curves(axes[1,0],table.summaries,'i2t_R1_percent','I2T: raw residual',predicate)
        plot_curves(axes[1,1],table.summaries,'t2i_R1_percent','T2I: raw residual',predicate)
        fig.suptitle('H1-1: retrieval in the cone subspace and its complement')
        save_figure(fig, table.png)
    return table.path, table.png


def pair_terms(z, t, image_indices, text_indices, frame):
    zi, ti = z[image_indices], t[text_indices]
    f = frame['image']
    delta = zi-f['mu']
    m = ti @ f['mu']
    v = (delta @ f['b'])*(ti @ f['b'])
    a = (delta @ f['e'])*(ti @ f['e'])
    # U를 독립적으로 계산해서 reconstruction을 검사한다.
    q = frame['spaces']['image_cone']
    u = np.sum(residual(delta, q)*residual(ti, q), axis=1)
    direct = np.sum(zi*ti,axis=1)
    error = np.abs(direct-m-v-a-u)
    if error.max() > 0.00001:
        raise AssertionError(f'네 항 재구성 오류: {error.max():.3f}')
    return dict(M=m, V=v, A_e=a, U=u, full=direct, reconstruction_error=error)


def selected_pair_margins(snapshot, query, direction, frame):
    rows = np.arange(len(query['positive']))
    if direction == 'i2t':
        p = pair_terms(snapshot.z,snapshot.t,rows,query['positive'],frame)
        n = pair_terms(snapshot.z,snapshot.t,rows,query['negative'],frame)
    else:
        p = pair_terms(snapshot.z,snapshot.t,query['positive'],rows,frame)
        n = pair_terms(snapshot.z,snapshot.t,query['negative'],rows,frame)
    return {key: p[key]-n[key] for key in ('M','V','A_e','U','full')}


def h1_exp2_margin_decomposition(snapshot, context, render=False):
    """H1-2. 현재 hard negative와 초기 고정 후보의 M/V/A_e/U 마진 및 변화."""
    terms = ('M','V','A_e','U','full')
    extra = ['direction','candidate_policy','status','n_queries','query_id','positive_candidate_id',
             'negative_candidate_id','initial_correct','current_correct','initial_rank','current_rank',
             'reconstruction_max_error','U_full_margin_pearson','matched_clip_rank','matched_clip_correct',
             'status_vs_clip_only','matched_clip_margin_available','matched_clip_full_margin',
             'matched_clip_U_margin','delta_full_vs_clip_only','delta_U_vs_clip_only']
    extra += [f'{term}_margin' for term in terms]+[f'delta_{term}_margin' for term in terms]
    table = context.table('h1_exp2_margin_decomposition', extra)
    current = query_stats(snapshot.scores,snapshot.mapping)
    control_key=snapshot.metadata['seed'],snapshot.metadata['epoch']
    is_control=snapshot.metadata['method']=='clip_only'
    if is_control:
        context.clip_queries[control_key]=current
    control_queries=context.clip_queries.get(control_key)
    for mode, frame in context.frames(snapshot):
        for direction in ('i2t','t2i'):
            oldq,newq = context.base_query[direction],current[direction]
            initial_correct,current_correct = oldq['ranks']==1,newq['ranks']==1
            status = np.where(initial_correct,np.where(current_correct,'retained','lost'),np.where(current_correct,'gained','still_wrong'))
            clipq=control_queries[direction] if control_queries is not None else None
            status_clip=(np.where(clipq['ranks']==1,np.where(current_correct,'retained_vs_clip','lost_vs_clip'),
                                 np.where(current_correct,'gained_vs_clip','still_wrong_vs_clip')) if clipq is not None else None)
            policies=['current_hard_negative','initial_fixed_candidates']
            if clipq is not None:policies.append('clip_only_fixed_candidates')
            for policy in policies:
                chosen = newq if policy=='current_hard_negative' else oldq if policy=='initial_fixed_candidates' else clipq
                margins = selected_pair_margins(snapshot,chosen,direction,frame)
                old = selected_pair_margins(context.initial,chosen,direction,context.initial.frame)
                key=(*control_key,mode,direction,'initial_fixed_candidates' if policy=='initial_fixed_candidates' else 'current_hard_negative')
                if is_control:
                    context.clip_margins[key]=margins
                # 자체 current-hard 후보는 CLIP 대조군의 후보와 달라 직접 차감하지 않는다.
                clip_margins=context.clip_margins.get(key) if policy!='current_hard_negative' or is_control else None
                error = float(np.max(np.abs(margins['full']-sum(margins[k] for k in terms[:-1]))))
                if direction=='t2i' and np.max(np.abs(margins['M'])) > EPS:
                    raise AssertionError('텍스트 고정 검색에서 M 상쇄가 실패했습니다.')
                meta = dict(**context.meta(snapshot,mode),direction=direction,candidate_policy=policy)
                groups=['all','retained','lost','gained','still_wrong']
                if status_clip is not None:groups+=['retained_vs_clip','lost_vs_clip','gained_vs_clip','still_wrong_vs_clip']
                for group in groups:
                    mask = np.ones(len(status),bool) if group=='all' else status_clip==group if group.endswith('_vs_clip') else status==group
                    if not mask.any(): continue
                    row = dict(**meta,record_type='summary',status=group,n_queries=int(mask.sum()),
                               reconstruction_max_error=error,U_full_margin_pearson=weighted_pearson(margins['U'][mask],margins['full'][mask]))
                    for term in terms:
                        row[f'{term}_margin']=float(margins[term][mask].mean())
                        row[f'delta_{term}_margin']=float((margins[term]-old[term])[mask].mean())
                    row['matched_clip_margin_available']=clip_margins is not None
                    if clip_margins is not None:
                        row.update(matched_clip_full_margin=float(clip_margins['full'][mask].mean()),
                                   matched_clip_U_margin=float(clip_margins['U'][mask].mean()),
                                   delta_full_vs_clip_only=float((margins['full']-clip_margins['full'])[mask].mean()),
                                   delta_U_vs_clip_only=float((margins['U']-clip_margins['U'])[mask].mean()))
                    table.append(row)
                if snapshot.metadata['epoch'] in context.cfg.query_detail_epochs:
                    query_ids = snapshot.image_ids if direction=='i2t' else snapshot.caption_ids
                    candidate_ids = snapshot.caption_ids if direction=='i2t' else snapshot.image_ids
                    for j, query_id in enumerate(query_ids):
                        row=dict(**meta,record_type='query',status=status[j],n_queries=1,query_id=int(query_id),
                                 positive_candidate_id=int(candidate_ids[chosen['positive'][j]]),
                                 negative_candidate_id=int(candidate_ids[chosen['negative'][j]]),
                                 initial_correct=bool(initial_correct[j]),current_correct=bool(current_correct[j]),
                                 initial_rank=int(oldq['ranks'][j]),current_rank=int(newq['ranks'][j]),reconstruction_max_error=error)
                        for term in terms:
                            row[f'{term}_margin']=float(margins[term][j])
                            row[f'delta_{term}_margin']=float(margins[term][j]-old[term][j])
                        row['matched_clip_margin_available']=clip_margins is not None
                        if clipq is not None:
                            row.update(matched_clip_rank=int(clipq['ranks'][j]),matched_clip_correct=bool(clipq['ranks'][j]==1),
                                       status_vs_clip_only=status_clip[j])
                        if clip_margins is not None:
                            row.update(matched_clip_full_margin=float(clip_margins['full'][j]),matched_clip_U_margin=float(clip_margins['U'][j]),
                                       delta_full_vs_clip_only=float(margins['full'][j]-clip_margins['full'][j]),
                                       delta_U_vs_clip_only=float(margins['U'][j]-clip_margins['U'][j]))
                        table.append(row)
    table.flush()
    if render:
        fig,axes=make_figure()
        for j,direction in enumerate(('i2t','t2i')):
            predicate=lambda r,d=direction:r['reference_mode']=='initial_fixed' and r['direction']==d and r['candidate_policy']=='initial_fixed_candidates' and r['status']=='all'
            plot_curves(axes[0,j],table.summaries,'delta_U_margin',f'{direction.upper()}: residual margin change',predicate)
            predicate=lambda r,d=direction:r['reference_mode']=='initial_fixed' and r['direction']==d and r['candidate_policy']=='clip_only_fixed_candidates' and r['status']=='lost_vs_clip'
            plot_curves(axes[1,j],table.summaries,'delta_U_vs_clip_only',f'{direction.upper()}: lost versus matched CLIP',predicate)
            axes[0,j].axhline(0,color='black',linewidth=0.8)
            axes[1,j].axhline(0,color='black',linewidth=0.8)
        fig.suptitle('H1-2: separate changing competitors from fixed-pair deformation')
        save_figure(fig,table.png)
    return table.path,table.png


def coordinate(x,frame,kind):
    c=x @ frame['r']
    if kind=='c': return c
    if kind=='C': return frame['R']*c
    return ((x-frame['b']) @ frame['a'])/frame['ell']


def h1_exp3_pair_coordinates(snapshot, context, render=False):
    """H1-3. 실제 정답 쌍의 C/c/h 대응과 이미지 단위 permutation 대조."""
    extra=['coordinate','n_pairs','weighting','paired_pearson','paired_mse','shuffled_pearson_mean',
           'shuffled_mse_mean','shuffled_mse_std','shuffle_advantage','image_coordinate_mean',
           'text_coordinate_mean','image_coordinate_std','text_coordinate_std']
    table=context.table('h1_exp3_pair_coordinates',extra)
    n=len(snapshot.z)
    # 캡션 수가 다른 이미지도 같은 총 가중치를 갖는다.
    weights=1/np.bincount(snapshot.mapping,minlength=n)[snapshot.mapping]
    weights=weights/weights.sum()
    for mode,frame in context.frames(snapshot):
        for kind in ('C','c','h'):
            xi=coordinate(snapshot.z,frame['image'],kind)
            xt=coordinate(snapshot.t,frame['text'],kind)
            paired=xi[snapshot.mapping]
            mse=float(weights @ ((paired-xt)**2))
            shuffled_mse=[];shuffled_corr=[]
            for trial in range(context.cfg.pair_shuffle_repeats):
                rng=np.random.default_rng(np.random.SeedSequence([context.cfg.control_seed,3101,trial]))
                perm=rng.permutation(n)
                shuffled=xi[perm[snapshot.mapping]]
                shuffled_mse.append(float(weights @ ((shuffled-xt)**2)))
                shuffled_corr.append(weighted_pearson(shuffled,xt,weights))
            xmean,ymean=float(weights @ paired),float(weights @ xt)
            row=dict(**context.meta(snapshot,mode),record_type='summary',coordinate=kind,n_pairs=len(xt),
                     weighting='equal_image_total_weight',paired_pearson=weighted_pearson(paired,xt,weights),paired_mse=mse,
                     shuffled_pearson_mean=float(np.nanmean(shuffled_corr)) if np.isfinite(shuffled_corr).any() else float('nan'),
                     shuffled_mse_mean=float(np.mean(shuffled_mse)),shuffled_mse_std=float(np.std(shuffled_mse,ddof=1)) if len(shuffled_mse)>1 else 0.000,
                     shuffle_advantage=float(np.mean(shuffled_mse)-mse),image_coordinate_mean=xmean,text_coordinate_mean=ymean,
                     image_coordinate_std=float(np.sqrt(weights @ ((paired-xmean)**2))),
                     text_coordinate_std=float(np.sqrt(weights @ ((xt-ymean)**2))))
            table.append(row)
    table.flush()
    if render:
        fig,axes=make_figure()
        for j,kind in enumerate(('c','h')):
            predicate=lambda r,k=kind:r['reference_mode']=='epoch_current' and r['coordinate']==k
            plot_curves(axes[0,j],table.summaries,'paired_pearson',f'{kind}: positive-pair correlation',predicate)
            plot_curves(axes[1,j],table.summaries,'shuffle_advantage',f'{kind}: shuffled MSE - paired MSE',predicate)
            axes[1,j].axhline(0,color='black',linewidth=0.8)
        fig.suptitle('H1-3: coordinate matching versus image-level shuffled correspondence')
        save_figure(fig,table.png)
    return table.path,table.png


def h1_exp4_residual_structure(snapshot, context, render=False):
    """H1-4. 기하/평면·잔차 분산/중심화 effective rank/초기 잔차 이웃 보존."""
    structural=['modality','joint_subspace_rank','total_variance','plane_variance','residual_variance',
                'plane_variance_fraction','axis_parallel_variance','axis_perpendicular_variance',
                'centered_effective_rank','centered_participation_ratio','residual_effective_rank',
                'residual_participation_ratio','residual_direction_cosine_to_initial',
                'residual_neighbor_overlap_to_initial','neighbor_k','i2t_failure_fraction','t2i_failure_fraction']
    table=context.table('h1_exp4_residual_structure',structural+GEOMETRY_COLUMNS+RETRIEVAL_COLUMNS)
    qstats=query_stats(snapshot.scores,snapshot.mapping)
    metrics=retrieval_summary(qstats)
    geom=geometry(snapshot.z,snapshot.t,snapshot.frame)
    pooled_t=text_per_image(snapshot.t,snapshot.mapping,len(snapshot.z))
    baseline_t=text_per_image(context.initial.t,context.initial.mapping,len(context.initial.z))
    for mode,frame in context.frames(snapshot):
        q=frame['spaces']['joint_cone']
        for modality,x,base,neighbors_x,neighbors_base in (
            ('image',snapshot.z,context.initial.z,snapshot.z,context.initial.z),
            ('text',snapshot.t,context.initial.t,pooled_t,baseline_t)):
            cov=covariance(x)
            projection=q @ q.T
            residual_cov=cov-projection @ cov-cov @ projection+projection @ cov @ projection
            var=float(np.trace(cov));plane_var=float(np.trace(q.T @ cov @ q))
            spectral=spectral_stats(cov);rspectral=spectral_stats(residual_cov)
            a=frame[modality]['a'];parallel=float(a @ cov @ a)
            # 잔차 방향 변화는 두 시점에 동일한 초기 기준 평면을 적용해 측정한다.
            fixed_q=context.initial.frame['spaces']['joint_cone']
            cur_fixed=unit(residual(x,fixed_q),allow_zero=True)
            base_fixed=unit(residual(base,fixed_q),allow_zero=True)
            good=(np.linalg.norm(cur_fixed,axis=1)>EPS)&(np.linalg.norm(base_fixed,axis=1)>EPS)
            drift=float(np.sum(cur_fixed[good]*base_fixed[good],axis=1).mean()) if good.any() else float('nan')
            # 현재 평면을 쓰는 이웃 비교도 두 시점에 같은 평면을 적용한다.
            nn=nearest_neighbors(residual(neighbors_x,q),context.cfg.neighbor_k)
            original_nn=nearest_neighbors(residual(neighbors_base,q),context.cfg.neighbor_k)
            row=dict(**context.meta(snapshot,mode),record_type='summary',modality=modality,
                     joint_subspace_rank=q.shape[1],total_variance=var,plane_variance=plane_var,
                     residual_variance=float(np.trace(residual_cov)),plane_variance_fraction=plane_var/var if var>EPS else float('nan'),
                     axis_parallel_variance=parallel,axis_perpendicular_variance=var-parallel,
                     centered_effective_rank=spectral['effective_rank'],centered_participation_ratio=spectral['participation_ratio'],
                     residual_effective_rank=rspectral['effective_rank'],residual_participation_ratio=rspectral['participation_ratio'],
                     residual_direction_cosine_to_initial=drift,residual_neighbor_overlap_to_initial=neighbor_overlap(nn,original_nn),
                     neighbor_k=min(context.cfg.neighbor_k,len(snapshot.z)-1),
                     i2t_failure_fraction=float(np.mean(qstats['i2t']['ranks']>1)),
                     t2i_failure_fraction=float(np.mean(qstats['t2i']['ranks']>1)),**geom,**metrics)
            if abs(var-(1-np.linalg.norm(x.mean(0))**2)) > 0.00001:
                raise AssertionError('단위 임베딩의 분산 항등식이 실패했습니다.')
            table.append(row)
    table.flush()
    if render:
        fig,axes=make_figure()
        for j,modality in enumerate(('image','text')):
            predicate=lambda r,m=modality:r['reference_mode']=='initial_fixed' and r['modality']==m
            plot_curves(axes[0,j],table.summaries,'residual_effective_rank',f'{modality}: centered residual effective rank',predicate)
            plot_curves(axes[1,j],table.summaries,'residual_neighbor_overlap_to_initial',f'{modality}: residual neighbor retention',predicate)
            axes[1,j].set_ylim(0,1.050)
        fig.suptitle('H1-4: preserve the initial reference when measuring residual deformation')
        save_figure(fig,table.png)
    return table.path,table.png



EXPERIMENTS = (h1_exp1_subspace_retrieval, h1_exp2_margin_decomposition,
               h1_exp3_pair_coordinates, h1_exp4_residual_structure)

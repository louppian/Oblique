"""Run one manuscript section and publish exactly one CSV plus one PNG."""
from pathlib import Path
from contextlib import contextmanager
from dataclasses import replace
import csv
import importlib
import tempfile
import shutil
import numpy as np
from PIL import Image

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'code').is_dir() and (p/'third_party').is_dir())


def publish(work, group, name):
    """Stream heterogeneous CSVs into one table; compose all figure panels."""
    files = sorted(Path(work).rglob('*.csv'))
    pictures = sorted(Path(work).rglob('*.png'))
    if not files or not pictures:
        raise RuntimeError(f'{name}: both CSV data and a figure are required')
    columns = ['analysis']
    for path in files:
        with path.open(encoding='utf-8-sig', newline='') as f:
            columns += [c for c in next(csv.reader(f)) if c not in columns]
    output = ROOT/'results'/group
    output.mkdir(parents=True, exist_ok=True)
    csv_temp = Path(work)/'_combined.csv.tmp'
    with csv_temp.open('w', encoding='utf-8-sig', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        for path in files:
            with path.open(encoding='utf-8-sig', newline='') as f:
                for row in csv.DictReader(f):
                    row['analysis'] = path.relative_to(work).with_suffix('').as_posix()
                    writer.writerow(row)
    sizes = []
    for path in pictures:
        with Image.open(path) as im: sizes.append(im.size)
    width = min(max(w for w,h in sizes), 2400)
    heights = [max(1,round(h*width/w)) for w,h in sizes]
    canvas = Image.new('RGB',(width,sum(heights)+24*(len(heights)-1)),'white')
    y = 0
    for path,height in zip(pictures,heights):
        with Image.open(path) as im:
            panel = im.convert('RGB').resize((width,height),Image.Resampling.LANCZOS)
            canvas.paste(panel,(0,y)); panel.close()
        y += height+24
    png_temp = Path(work)/'_combined.png.tmp'
    canvas.save(png_temp,format='PNG');canvas.close()
    csv_temp.replace(output/f'{name}.csv')
    png_temp.replace(output/f'{name}.png')
    print(f'Completed: {output/name}.csv / .png',flush=True)
    return output/f'{name}.csv',output/f'{name}.png'


@contextmanager
def workspace(group,name):
    cache = ROOT/'cache'/'work'
    cache.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=name+'_',dir=cache) as tmp:
        yield Path(tmp)
        publish(Path(tmp),group,name)


def run_existing(group,name,module,functions,config):
    an = importlib.import_module('1_analysis')
    target = importlib.import_module(module)
    with workspace(group,name) as work:
        original = an.result_directory
        an.result_directory = lambda root,section: work/section
        try:
            if module == '3_alignment':
                for fn in functions: getattr(target,fn)(config)
            else:
                ctx = an.Context(config,out_name=module)
                try:
                    for fn in functions: getattr(target,fn)(ctx)
                finally:
                    for encoder in ctx._encoders.values():
                        if hasattr(encoder,'close'): encoder.close()
        finally:
            an.result_directory = original


def run_retrieval(group,name,config):
    import residual_core as core
    with workspace(group,name) as work:
        core.main(replace(config,output_directory=work/'retrieval'))


def run_medical_training(group,name,config):
    """Existing text-encoder alignment training; checkpoints stay under cache."""
    import train_negation_alignment as training
    import pandas as pd
    import matplotlib.pyplot as plt
    destination = ROOT/'cache'/'residual_alignment_training'
    destination.mkdir(parents=True,exist_ok=True)
    before = set(destination.iterdir())
    original = training.TrainConfig
    training.TrainConfig = lambda: config
    try:
        training.train()
    finally:
        training.TrainConfig = original
    created = set(destination.iterdir())-before
    if len(created)!=1: raise RuntimeError('Expected one completed training directory')
    completed = created.pop()
    with workspace(group,name) as work:
        for path in completed.glob('*.csv'): shutil.copy2(path,work/path.name)
        frame = pd.read_csv(completed/'test_comparison.csv')
        columns = [c for c in ('auc_before','auc_after') if c in frame]
        fig,ax=plt.subplots(figsize=(12,5))
        frame.groupby('disease')[columns].mean().plot.bar(ax=ax)
        ax.set(ylabel='AUC',title='Held-out disease alignment evaluation')
        fig.tight_layout();fig.savefig(work/'training.png',dpi=180);plt.close(fig)

"""Download the official gated validation split using existing authorized HF login."""
from pathlib import Path
import json
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download
ROOT=Path(__file__).resolve().parents[2]/'data/imagenet'

def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    snapshot_download('ILSVRC/imagenet-1k',repo_type='dataset',token=True,allow_patterns=['data/validation-*.parquet'],local_dir=ROOT/'source',max_workers=3)
    paths=sorted((ROOT/'source/data').glob('validation-*.parquet'))
    assert len(paths)==14
    counts=[0]*1000;total=0
    for path in paths:
        table=pq.ParquetFile(path)
        names=json.loads(table.schema_arrow.metadata[b'huggingface'])['info']['features']['label']['names']
        assert len(names)==1000
        for batch in table.iter_batches(batch_size=128):
            data=batch.to_pydict()
            for image,label in zip(data['image'],data['label']):
                folder=ROOT/'val'/f'{label:04d}';folder.mkdir(exist_ok=True,parents=True)
                target=folder/f'{total:08d}.jpg'
                if not target.exists():target.write_bytes(image['bytes'])
                counts[label]+=1;total+=1
        print('Extracted',total,flush=True)
    assert total==50000 and counts==[50]*1000
    (ROOT/'metadata.json').write_text(json.dumps(dict(source='ILSVRC/imagenet-1k',split='validation',count=total,class_names=names,class_counts=counts),ensure_ascii=False,indent=2),encoding='utf-8')
    print('ImageNet validation ready',ROOT,flush=True)

if __name__=='__main__':main()

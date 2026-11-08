"""Locate selected source using ONLY ID columns, then read one matching row group."""
import ast
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from src.training.runtime_v2 import sha,write
from src.data.screen_code_interfaces import python_code

OUT=Path('reports/code_training_repair_20261025_v4')


def main():
    candidate=json.loads((OUT/'candidate_private.json').read_text(encoding='utf-8'))
    metadata=json.loads(Path('data/screening_sources/code_source.json').read_text(encoding='utf-8'))
    assert metadata['revision']==candidate['source_revision']
    wanted=candidate['sample_id'].removeprefix('code/')
    evidence=None
    searched=[]
    for file in metadata['files']:
        path=Path(file.get('path','data/screening_sources/code/'+file['name']))
        pf=pq.ParquetFile(path)
        searched.append(path.as_posix())
        for group in range(pf.num_row_groups):
            ids=pf.read_row_group(group,columns=['problem_id']).column(0).to_pylist()
            if wanted not in ids:continue
            offset=ids.index(wanted)
            table=pf.read_row_group(group,columns=['problem_id','gold_standard_solution','verification_info'])
            raw=table.slice(offset,1).to_pylist()[0]
            raw_code=raw['gold_standard_solution']
            normalized_code,tree=python_code(raw_code)
            raw_ast=hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()
            assert raw_ast==candidate['reference_ast_sha256']
            assert normalized_code==candidate['reference_code']
            assert raw['verification_info']==candidate['verification_info']
            actual_digest=sha(path)
            assert actual_digest==file['sha256']
            evidence={'repo':metadata['repo'],'revision':metadata['revision'],'source_file':path.as_posix(),
                'source_file_sha256':actual_digest,'row_group':group,'offset_in_row_group':offset,
                'problem_id':wanted,'ast_sha256':raw_ast,'reference_and_tests_byte_content_match':True,
                'raw_source_code_sha256':hashlib.sha256(raw_code.encode()).hexdigest(),
                'normalized_code_sha256':hashlib.sha256(normalized_code.encode()).hexdigest(),
                'normalization':'Existing python_code fence extraction only; candidate text unchanged',
                'files_id_columns_checked':searched,
                'scope':'Only ID columns searched; selected row-group code/tests verified; no downloads'}
            break
        if evidence:break
    if evidence is None:raise ValueError('Selected source problem_id not found')
    write(OUT/'source_verification.json',evidence)
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

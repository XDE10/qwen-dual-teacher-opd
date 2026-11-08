"""CPU Docker validation: full source tests, independent oracle, wrong programs."""
import json
from pathlib import Path
import random

from src.data.code_sandbox_v2 import run_jobs, environment
from src.training.runtime_v2 import sha, write

OUT=Path('reports/code_training_repair_20261025_v4')


def range_oracle_cases(seed=20261025, count=48):
    rng=random.Random(seed)
    cases=[]
    for i in range(count):
        n=1 if i==0 else rng.randint(2,14)
        values=[0]*n
        commands=[]
        outputs=[]
        # Overlapping sets, resetting to zero, negatives, and a single-cell query.
        operations=[(0,0,n-1,5),(0,0,n-1,0),(1,0,n-1),
                    (0,0,0,-3),(1,0,0)]
        for _ in range(40):
            s=rng.randrange(n);t=rng.randrange(s,n)
            operations.append((0,s,t,rng.randint(-20,20)) if rng.random()<.6 else (1,s,t))
        for operation in operations:
            commands.append(' '.join(map(str,operation)))
            if operation[0]==0:
                _,s,t,x=operation
                values[s:t+1]=[x]*(t-s+1)
            else:
                _,s,t=operation
                outputs.append(str(sum(values[s:t+1])))
        cases.append({'input':f'{n} {len(commands)}\n'+'\n'.join(commands)+'\n',
                      'output':'\n'.join(outputs)+'\n'})
    return cases


def subarray_oracle_cases(seed=20261025, count=48):
    rng=random.Random(seed)
    result=[]
    for i in range(count):
        n=rng.randint(1,22);m=rng.randint(2,17)
        array=[rng.randint(1,50) for _ in range(n)]
        if i==0:n,m,array=3,2,[1,1,1]
        if i==1:n,m,array=1,2,[2]
        if i==2:n,m,array=1,2,[1]
        if i==3:n,m,array=8,3,[3]*8
        total=0
        for left in range(n):
            value=0
            for right in range(left,n):
                value+=array[right]
                total+=value%m==0
        result.append({'input':f'{n} {m}\n'+ ' '.join(map(str,array))+'\n','output':str(total)+'\n'})
    return result


def oracle_cases(seed=20261025,count=48):
    from itertools import permutations
    rng=random.Random(seed)
    result=[]
    for i in range(count):
        examples=[(1,2,3,4),(5,5,5,5),(3,1,4,1),(100,20,20,100)]
        examples += [tuple(rng.randint(1,50) for _ in range(4)) for _ in range(8)]
        answer=[max(min(p[:2])*min(p[2:]) for p in permutations(a)) for a in examples]
        result.append({'input':str(len(examples))+'\n'+'\n'.join(' '.join(map(str,a)) for a in examples)+'\n',
                       'output':'\n'.join(map(str,answer))+'\n'})
    return result


def main():
    row=json.loads((OUT/'candidate_private.json').read_text(encoding='utf-8'))
    source=row['verification_info']['test_cases']
    independent=oracle_cases()
    write(OUT/'independent_oracle_cases.json',{'seed':20261025,'cases':independent,
          'method':'Host trusted enumeration of all 24 assignments to opposite-side pairs; rectangle area bounded by each pair minimum; source code ONLY inside Docker'})
    def job(name,code,cases):
        return {'sample_id':name,'interface':'stdin_stdout','candidate_code':code,
                'tests':cases,'comparator':{'type':'stdout_tokens_exact'}}
    wrong_area="""import sys
it=iter(map(int,sys.stdin.buffer.read().split()))
t=next(it)
for _ in range(t):
 a=sorted(next(it) for _ in range(4))
 print(a[0]*a[1])
"""
    jobs=[job('source_all',row['reference_code'],source),
          job('independent_oracle_48',row['reference_code'],independent),
          job('mutant_constant_first_output','import sys\nsys.stdout.write('+repr(source[0]['output'])+')',source),
          job('mutant_empty_output','pass',source),
          job('mutant_wrong_side_pair',wrong_area,independent)]
    worker=Path('scripts/code_eval_worker_v3.py')
    environment(worker=worker,record_path=OUT/'sandbox_environment.json')
    result=run_jobs(jobs,worker=worker)
    result.update(sample_id=row['sample_id'],source_count=len(source),independent_count=len(independent),
                  candidate_private_sha256=sha(OUT/'candidate_private.json'),worker_sha256=sha(worker),
                  scope='Provided tests plus independent small cases; not exhaustive proof or model score')
    write(OUT/'execution.json',result)
    by_id={r['sample_id']:r for r in result['results']}
    assert by_id['source_all']['status']=='passed' and by_id['source_all']['checked']==len(source)
    assert by_id['independent_oracle_48']['status']=='passed' and by_id['independent_oracle_48']['checked']==len(independent)
    assert all(by_id[n]['status']=='mismatch' for n in ['mutant_constant_first_output','mutant_empty_output','mutant_wrong_side_pair'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

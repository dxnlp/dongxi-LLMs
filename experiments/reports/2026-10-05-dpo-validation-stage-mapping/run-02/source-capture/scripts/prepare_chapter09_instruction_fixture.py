"""Generate original English interface tasks with disjoint value/source groups."""
import argparse
import hashlib
import json
from pathlib import Path


def records(split,start,count):
    rows=[]
    for value in range(start,start+count):
        first,second=f'item{value}',f'code{value}'
        tasks=[('copy',f'Reply with exactly this word: {first}',first),
               ('reverse',f'Reverse these two words, separated by one space: {first} {second}',f'{second} {first}'),
               ('extract',f'Return only the value after label. label={second}; extra={first}',second)]
        for family,prompt,answer in tasks:
            rows.append(dict(id=f'{split}-{value}-{family}',group=f'value-{value}',split=split,
                             family=family,source='original deterministic course generator',
                             messages=[dict(role='system',content='Follow the requested output format exactly.'),
                                       dict(role='user',content=prompt),dict(role='assistant',content=answer)]))
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Generate into a new empty directory')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest={}
    for split,start,count in [('train',0,80),('dev',80,20),('test',100,40)]:
        rows=records(split,start,count)
        content=''.join(json.dumps(row,sort_keys=True)+'\n' for row in rows)
        (args.output/f'{split}.jsonl').write_text(content)
        manifest[split]=dict(examples=len(rows),source_groups=count,sha256=hashlib.sha256(content.encode()).hexdigest())
    card=dict(name='Original instruction interface v1',creator='Dongxi LLMs course',
              license='Original course-authored fixture; redistribution follows the repository license when declared; no upstream material copied',
              tasks=['copy','reverse two words','extract labeled value'],splits=manifest,
              split_rule='disjoint numeric value groups, all task families stay together',
              limitations=['narrow deterministic tasks','held-out lexical values, shared task templates',
                           'not a general assistant benchmark','publication test never used for recipe selection'],
              mask_policy='assistant body plus end-of-message delimiter and trailing template separator',
              truncation='reject overlength records; no silent truncation')
    (args.output/'data-card.json').write_text(json.dumps(card,indent=2)+'\n')
    print(json.dumps(card,indent=2))


if __name__=='__main__':
    main()

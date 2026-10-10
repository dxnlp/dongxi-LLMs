"""One-pass local integration audit for this foundations revision.

Reuses the course's existing editorial local-target and solution-ID review
recipes. This records source preservation and navigation; content review and
actual CPU execution are separate evidence. No historical receipt is rewritten.
"""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess
import urllib.parse

ROOT=Path(__file__).resolve().parents[4]
RUN=Path(__file__).resolve().parent
baseline=json.loads((RUN/'baseline.json').read_text())
base=baseline['base_commit']
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def original(path):
    return subprocess.check_output(['git','show',f'{base}:{path}'],cwd=ROOT,text=True)
def unfence(source):
    return re.sub(r'(?ms)^\s*(`{3,}|~{3,}).*?^\s*\1[^\n]*$','',source)
def anchors(source):
    result=set(re.findall(r'<a\s+id="([^"]+)"',source));counts={}
    for match in re.finditer(r'(?m)^#{1,6}\s+(.+)$',unfence(source)):
        title=re.sub(r'\[([^]]+)\]\([^)]+\)',r'\1',match[1].strip().rstrip('#').strip()).lower()
        slug=''.join(c for c in title if c.isalnum() or c in (' ','-','_')).replace(' ','-')
        count=counts.get(slug,0);counts[slug]=count+1
        result.add(slug if not count else f'{slug}-{count}')
    return result

def inventory(source):
    return {'lines':len(source.splitlines()),'raw_markdown_tokens':len(source.split()),
            'figure_links':len(re.findall(r'!\[[^\]]*\]\(',source))}

protected_errors=[path for path,digest in baseline['protected_hashes'].items()
                  if not (ROOT/path).exists() or sha(ROOT/path)!=digest]
links=0;link_issues=[]
scope=sorted((ROOT/'book').rglob('*.md'))+sorted((ROOT/'docs/runbooks').glob('*.md'))+[
    ROOT/'docs/BOOK_FOUNDATIONS_PLAN.md',ROOT/'experiments/reports/2026-10-10-book-foundations-pass.md']
for path in scope:
    for match in re.finditer(r'\]\(([^\s)]+)\)',unfence(path.read_text())):
        target=match[1]
        if urllib.parse.urlsplit(target).scheme: continue
        name,_,fragment=target.partition('#')
        dest=(path.parent/urllib.parse.unquote(name)).resolve() if name else path
        links+=1
        if not dest.exists(): link_issues.append({'path':str(path.relative_to(ROOT)),'target':target,'error':'missing file'})
        elif fragment and dest.suffix=='.md' and urllib.parse.unquote(fragment) not in anchors(dest.read_text()):
            link_issues.append({'path':str(path.relative_to(ROOT)),'target':target,'error':'missing heading'})

chapters=[];new_code=[];syntax_issues=[]
for path in sorted((ROOT/'book/chapters').glob('*.md')):
    rel=str(path.relative_to(ROOT));before=original(rel);after=path.read_text()
    old_questions=re.findall(r'(?m)^\d+\. .+$',unfence(before))
    new_questions=re.findall(r'(?m)^\d+\. .+$',unfence(after))
    missing=[line for line in old_questions if line not in new_questions]
    chapters.append({'path':rel,'before_sha256':baseline['book_hashes'][rel],'after_sha256':sha(path),
                     'before':inventory(before),'after':inventory(after),
                     'original_numbered_question_lines':len(old_questions),'missing_original_question_lines':missing})
    old_blocks=re.findall(r'(?ms)^```python\s*\n(.*?)^```\s*$',before)
    for block in re.findall(r'(?ms)^```python\s*\n(.*?)^```\s*$',after):
        if block in old_blocks: continue
        record={'chapter':rel,'sha256':hashlib.sha256(block.encode()).hexdigest(),'syntax':'passed'}
        try: ast.parse(block)
        except SyntaxError as error:
            record['syntax']='failed';record['error']=str(error);syntax_issues.append(record)
        new_code.append(record)

solutions=[]
for path in sorted((ROOT/'book/solutions').glob('*.md')):
    rel=str(path.relative_to(ROOT));before=original(rel);after=path.read_text()
    pattern=r'^#{2,3} (?:Exercise )?([0-9]+)\b'
    old=re.findall(pattern,before,re.M);new=re.findall(pattern,after,re.M)
    solutions.append({'path':rel,'original_ids':old,'current_ids':new,'preserved':old==new,'sha256':sha(path)})
issues=protected_errors+link_issues+syntax_issues+[r for r in chapters if r['missing_original_question_lines']]+[r for r in solutions if not r['preserved']]
record={'schema_version':1,'scope':'Final preservation, inventory, navigation, numbered exercise/solution and new listing syntax audit; independent content and execution reviewed separately',
        'base_commit':base,'protected_files':len(baseline['protected_hashes']),'protected_errors':protected_errors,
        'local_targets_checked':links,'local_target_issues':link_issues,'chapters':chapters,'solutions':solutions,
        'new_python_listings':new_code,'status':'passed' if not issues else 'failed'}
(RUN/'integration-check.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':record['status'],'protected_files':record['protected_files'],'protected_errors':protected_errors,
                  'local_targets_checked':links,'local_target_issues':link_issues,'changed_chapters':sum(r['before_sha256']!=r['after_sha256'] for r in chapters),
                  'original_numbered_solution_headings':sum(len(r['original_ids']) for r in solutions),'new_listings':len(new_code),'issues':issues},indent=2))
raise SystemExit(bool(issues))

#!/usr/bin/env python3
"""Exercise the recorded selection fragment in an installed local CPU browser.

No download, server or network dependency. Output is new evidence, never an
overwrite. Browser tests measure DOM/control behavior, not learner mastery.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import subprocess

TEST = r"""
<pre id="dxi-verification" hidden></pre>
<script>
(() => {
  const root=document.getElementById('dxi-selection-budget');
  const count=root.querySelector('#dxi-count'), rule=root.querySelector('#dxi-rule');
  const answers=['1','0','0','1','1','1','1','0'];
  const checks=[];
  for (let n=1;n<=8;n++) {
    for (const method of ['first','majority','mean_logp']) {
      count.value=n; count.dispatchEvent(new Event('input',{bubbles:true}));
      rule.value=method; rule.dispatchEvent(new Event('change',{bubbles:true}));
      const pool=answers.slice(0,n), zeros=pool.filter(x=>x==='0').length;
      const expected=method==='majority' && zeros>n-zeros ? '0' : '1';
      checks.push({n,method,
        selected:root.querySelector('#dxi-selected').textContent.startsWith(expected+' · '),
        available:root.querySelector('#dxi-available').textContent===(zeros?'Yes':'No'),
        cost:root.querySelector('#dxi-cost').textContent===`${2*n} + ${2*n}`,
        active:[...root.querySelectorAll('.dxi-candidates li')].filter(x=>x.dataset.active==='true').length===n});
    }
  }
  window.dispatchEvent(new CustomEvent('openai:set_globals',{detail:{globals:{widgetState:{modelContent:{n:3,rule:'majority'}}}}}));
  const restored=count.value==='3' && rule.value==='majority' && root.querySelector('#dxi-selected').textContent==='0 · correct';
  window.dispatchEvent(new CustomEvent('openai:set_globals',{detail:{globals:{widgetState:{modelContent:{n:99,rule:'bad'}}}}}));
  const badStateFallback=count.value==='8' && rule.value==='majority';
  const box=root.getBoundingClientRect();
  const overflowing=[...root.querySelectorAll('*')].filter(x=> {
    const b=x.getBoundingClientRect();return b.width>0 && (b.left<box.left-1 || b.right>box.right+1);
  }).map(x=>x.id||x.tagName);
  const passed=checks.every(x=>x.selected&&x.available&&x.cost&&x.active)&&restored&&badStateFallback&&!overflowing.length;
  document.getElementById('dxi-verification').textContent=JSON.stringify({passed,width:innerWidth,checks,restored,badStateFallback,overflowing});
})();
</script>
"""


class ResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active=False
        self.value=[]

    def handle_starttag(self,tag,attrs):
        if tag=='pre' and dict(attrs).get('id')=='dxi-verification':
            self.active=True

    def handle_endtag(self,tag):
        if tag=='pre': self.active=False

    def handle_data(self,data):
        if self.active: self.value.append(data)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fragment',type=Path,required=True)
    parser.add_argument('--stylesheet',type=Path,required=True)
    parser.add_argument('--browser',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Unused evidence directory required')
    root=Path(__file__).resolve().parents[1]
    fragment=args.fragment.read_text()
    if any(x in fragment.lower() for x in ('<!doctype','<html','<head','<body','fetch(','websocket','xmlhttprequest')):
        parser.error('Offline fragment required')
    # Verify the displayed measurement against its original response pool.
    raw=root/'experiments/reports/2026-10-04-inference-selection/results.json'
    result=json.loads(raw.read_text())
    run=next(x for x in result['runs'] if x['seed']==10052)
    policy=next(x for x in run['policies'] if x['label']=='sft-final')
    pool=next(x for x in policy['pools'] if x['item_id']=='train-11')
    records=pool['raw_candidates']
    assert [x['response_text'] for x in records]==['1','0','0','1','1','1','1','0']
    assert all(x['generated_tokens']==2 and x['cost']['scoring_tokens']==2 for x in records)
    assert all(x['sample_id'] in fragment and repr(sum(x['rescored_log_probabilities'])/len(x['token_ids'])) in fragment for x in records)
    item=next(x for x in json.loads((root/'fixtures/inference-selection/items.json').read_text()) if x['id']=='train-11')
    assert item['reference']=='0'
    args.output.mkdir(parents=True)
    doc=args.output/'browser-test.html'
    doc.write_text('<!doctype html><html><head><meta charset="utf-8"><style>'+args.stylesheet.read_text()+
        '</style></head><body>'+fragment+TEST+'</body></html>')
    browser_version=subprocess.run([str(args.browser),'--version'],capture_output=True,text=True,check=True).stdout.strip()
    tests=[]
    for width in (320,736):
        command=[str(args.browser),'--no-sandbox','--disable-gpu','--disable-dev-shm-usage',
            '--hide-scrollbars',f'--window-size={width},1000','--virtual-time-budget=1500',
            '--dump-dom',f'--screenshot={str((args.output/f"width-{width}.png").resolve())}',doc.resolve().as_uri()]
        process=subprocess.run(command,capture_output=True,text=True,timeout=30)
        (args.output/f'width-{width}.dom.html').write_text(process.stdout)
        (args.output/f'width-{width}.stderr.txt').write_text(process.stderr)
        parsed=ResultParser(); parsed.feed(process.stdout)
        try: row=json.loads(''.join(parsed.value))
        except ValueError: row={'passed':False,'error':'Browser test result absent/invalid'}
        row.update(exit_code=process.returncode,command=command)
        tests.append(row)
    evidence={'date_utc':datetime.now(timezone.utc).isoformat(),'status':'passed' if all(t['passed'] and t['exit_code']==0 for t in tests) else 'failed',
        'fragment':str(args.fragment),'fragment_sha256':digest(args.fragment),'raw_measurement_sha256':digest(raw),
        'verifier_sha256':digest(Path(__file__)),'browser':browser_version,'tests':tests,
        'scope':'Installed local CPU browser, real input/change events and restored state; no inline-host rendering, network or learner assessment claim'}
    (args.output/'verification.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'status':evidence['status'],'output':str(args.output),'widths':[t.get('width') for t in tests]}))
    if evidence['status']!='passed': raise SystemExit(1)


if __name__=='__main__': main()

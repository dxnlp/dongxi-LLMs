"""Original bounded sensitivity reference; reuse the course fit, no training."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json,math,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dongxi_llms.scaling_lab import simulated_rows,fit_isoflops,freeze_prediction,evaluate_holdout,ScalingFitError
out=Path(__file__).with_name('scaling-sensitivity-01');out.mkdir(exist_ok=False)
spec={'mode':'authored-simulation-only','calibration_budgets':[6e6,12e6,24e6,96e6],'seeds':[6601,6602,6603],'noise':.015,'heldout_budget':384e6,'runtime_seconds_cap':30,'conditions':['all-four-budgets','lower-three-budgets','upper-three-budgets','drop-seed6603','omit-analytic-center-sizes','assumed-floor1.9'],'prediction':'Range, point removal and assumed floor may alter the fitted exponent/prediction. Preserve every refusal; no condition is selected using heldout errors.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(out/'frozen-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
start=time.monotonic();cal=simulated_rows(budgets=spec['calibration_budgets'],seeds=spec['seeds'],noise=spec['noise'])
conditions=[('all-four-budgets',cal,2.),('lower-three-budgets',[r for r in cal if r['compute_budget']<=24e6],2.),('upper-three-budgets',[r for r in cal if r['compute_budget']>=12e6],2.),('drop-seed6603',[r for r in cal if r['seed']!=6603],2.),('omit-analytic-center-sizes',[r for r in cal if r['parameters']!=round(math.sqrt(r['compute_budget']/6))],2.),('assumed-floor1.9',cal,1.9)]
results=[]
for name,rows,floor in conditions:
 try:
  law=fit_isoflops(rows,assumed_loss_floor=floor);frozen=freeze_prediction(law,spec['heldout_budget']);result={'condition':name,'status':'predicted','law':asdict(law),'frozen':frozen}
 except ScalingFitError as e:result={'condition':name,'status':'refused','reason':str(e),'diagnostics':e.diagnostics}
 results.append(result)
(out/'frozen-all-predictions.json').write_text(json.dumps(results,indent=2)+'\n')
heldout=simulated_rows(budgets=(spec['heldout_budget'],),seeds=spec['seeds'],noise=spec['noise'])
for result in results:
 if result['status']=='predicted':result['holdout']=evaluate_holdout(result['frozen'],heldout)
fig,ax=plt.subplots(1,2,figsize=(12,5));ok=[r for r in results if r['status']=='predicted'];names=[r['condition'] for r in ok]
ax[0].barh(names,[r['law']['parameter_exponent'] for r in ok]);ax[0].axvline(.5,color='black',linestyle='--',label='Authored noiseless exponent');ax[0].set_xlabel('Fitted optimal-parameter exponent');ax[0].legend()
ax[1].barh(names,[r['holdout']['parameter_relative_error'] for r in ok]);ax[1].axvline(0,color='black');ax[1].set_xlabel('Signed predicted/grid-optimum relative error');fig.suptitle('Declared sensitivity conditions; authored simulation, no recipe selection');fig.tight_layout();fig.savefig(out/'sensitivity.png',dpi=140);plt.close(fig)
receipt={'spec':spec,'calibration':cal,'results':results,'heldout_rows':heldout,'runtime_seconds':time.monotonic()-start,'boundary':'Same authored surface with fixed replicate noise; range/point/floor sensitivity only. Every prediction frozen before generating target rows. No empirical LM law or best-condition selection.'}
if receipt['runtime_seconds']>30:raise TimeoutError('Frozen sensitivity cap exceeded')
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print([(r['condition'],r['status'],r.get('law',{}).get('parameter_exponent')) for r in results])

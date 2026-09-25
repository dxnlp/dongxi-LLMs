'use strict';
(() => {
  const finite = value => typeof value === 'number' && Number.isFinite(value);
  const number = (value, digits = 2) => finite(value) ? value.toLocaleString('en', {maximumFractionDigits: digits}) : '—';
  const seriesPoints = (rows, x, y) => rows.map(row => ({x:x(row), y:y(row), row})).filter(p => finite(p.x) && finite(p.y));
  const utilization = rows => {
    const valid = rows.reduce((s,r) => s + (finite(r.valid_targets) ? r.valid_targets : 0), 0);
    const positions = rows.reduce((s,r) => s + (finite(r.processed_positions) ? r.processed_positions : 0), 0);
    return positions > 0 ? valid / positions : null;
  };
  globalThis.monitorMath = {finite, number, seriesPoints, utilization};
  if (typeof document === 'undefined') return;
  const $ = id => document.getElementById(id);
  let state = null, busy = false, charts = [];
  const fmtAxis = value => {
    if (Math.abs(value) >= 1000000) return (value/1000000).toPrecision(3)+'M';
    if (Math.abs(value) >= 1000) return (value/1000).toPrecision(3)+'k';
    if (value !== 0 && Math.abs(value) < .01) return value.toExponential(1);
    return number(value, 2);
  };
  function options(select, choices, fallback) {
    const prior = select.value;
    select.replaceChildren(...choices.map(({value, label}) => {
      const option = document.createElement('option'); option.value = value; option.textContent = label; return option;
    }));
    select.value = choices.some(c => c.value === prior) ? prior : (fallback ?? choices[0]?.value ?? '');
  }
  function chart(id, groups, xLabel, yLabel, reference = null) {
    const canvas = $(id), context = canvas.getContext('2d');
    const width = canvas.clientWidth, height = canvas.clientHeight, ratio = devicePixelRatio || 1;
    canvas.width = width * ratio; canvas.height = height * ratio; context.scale(ratio, ratio);
    const theme = getComputedStyle(document.documentElement);
    const color = name => theme.getPropertyValue(name).trim();
    const colors = [color('--accent'), color('--second')];
    const all = groups.flatMap(g => g.points), left = 64, right = width-15, top = 18, bottom = height-51;
    context.font = '12px system-ui'; context.fillStyle = color('--muted');
    if (!all.length) {context.fillText('No recorded measurements yet', left, height/2); canvas.onpointermove=null;return;}
    let x0=Math.min(...all.map(p=>p.x)), x1=Math.max(...all.map(p=>p.x));
    let y0=Math.min(...all.map(p=>p.y)), y1=Math.max(...all.map(p=>p.y));
    if(finite(reference)){y0=Math.min(y0,reference);y1=Math.max(y1,reference);}
    if(x0===x1){x0-=.5;x1+=.5;} const padding=(y1-y0 || Math.abs(y1)*.1 || 1)*.12;y0-=padding;y1+=padding;
    const sx=x=>left+(x-x0)/(x1-x0)*(right-left),sy=y=>bottom-(y-y0)/(y1-y0)*(bottom-top);
    context.strokeStyle=color('--line');context.lineWidth=1;
    for(let i=0;i<4;i++){
      const y=y0+(y1-y0)*i/3,py=sy(y);context.beginPath();context.moveTo(left,py);context.lineTo(right,py);context.stroke();
      context.textAlign='right';context.fillText(fmtAxis(y),left-9,py+4);
    }
    for(let i=0;i<3;i++){
      const x=x0+(x1-x0)*i/2;context.textAlign=i===0?'left':i===2?'right':'center';context.fillText(fmtAxis(x),sx(x),bottom+20);
    }
    context.textAlign='center';context.fillText(xLabel,(left+right)/2,height-8);
    context.save();context.translate(14,(top+bottom)/2);context.rotate(-Math.PI/2);context.fillText(yLabel,0,0);context.restore();
    if(finite(reference)){
      context.strokeStyle=colors[1];context.setLineDash([4,4]);context.beginPath();context.moveTo(left,sy(reference));context.lineTo(right,sy(reference));context.stroke();context.setLineDash([]);
    }
    groups.forEach((group,i)=>{
      context.strokeStyle=colors[i%2];context.fillStyle=colors[i%2];context.lineWidth=2;context.beginPath();
      group.points.forEach((p,j)=>{if(j===0)context.moveTo(sx(p.x),sy(p.y));else context.lineTo(sx(p.x),sy(p.y));});context.stroke();
      if(group.points.length<100) group.points.forEach(p=>{context.beginPath();context.arc(sx(p.x),sy(p.y),3,0,Math.PI*2);context.fill();});
    });
    canvas.onpointermove = event => {
      const px=event.clientX-canvas.getBoundingClientRect().left;
      const nearest=groups.filter(g=>g.points.length).map(g=>{
        const point=g.points.reduce((a,b)=>Math.abs(sx(a.x)-px)<Math.abs(sx(b.x)-px)?a:b);
        return `${g.label}: ${number(point.y,6)} (${point.row.component || xLabel+' '+number(point.x)})`;
      });canvas.title=nearest.join('\n');
    };
  }
  function draw() {charts.forEach(args=>chart(...args));}
  function stories() {
    if(!state)return;
    for(const side of ['before','after']){
      const snapshot=state.snapshots.find(s=>s.name===$(side).value);
      const sample=snapshot?.samples?.find(s=>s.prompt===$('prompt').value&&s.method===$('method').value);
      $('story-'+side).textContent=sample?.continuation ?? 'No matching completion recorded.';
      $('ending-'+side).textContent=sample ? `${sample.token_ids?.length ?? '?'} generated tokens · ${sample.ended_with_eos?'EOS emitted':'No EOS in recorded continuation'} · limit ${sample.max_new_tokens ?? '?'} · seed ${sample.seed ?? '?'}` : '';
    }
  }
  function render(data) {
    state=data;$('content').hidden=false;$('state').textContent=data.state;
    const model=data.config.contract?.model||{},recipe=data.config.contract?.recipe||{},rows=data.metrics,last=rows.at(-1)||{};
    const latest=data.snapshots.at(-1),val=latest?.validation;
    $('identity').textContent=`${model.layers ?? '?'} blocks · width ${model.width ?? '?'} · context ${model.max_length ?? '?'} · ${data.config.contract?.device ?? 'unknown device'}`;
    $('warnings').replaceChildren(...data.warnings.map(message=>{const p=document.createElement('p');p.className='warning';p.textContent=message;return p;}));
    $('updates').textContent=number(last.update ?? latest?.update,0);$('budget').textContent=`Schedule horizon: ${number(recipe.total_updates,0)} updates`;
    $('loss').textContent=number(val?.loss,4);$('validation-scope').textContent=val?`${number(val.valid_targets,0)} targets · ${number(val.windows,0)} windows · ${val.complete_prepared_split?'prepared split':'partial split'}`:'Not recorded';
    $('speed').textContent=number(last.valid_targets_per_second,0);
    $('memory').textContent=data.memory.length?number(data.memory.at(-1).available_gib,1)+' GiB':'—';
    $('memory-detail').textContent=`Reserve: ${number(data.reserve_gib)} GiB · last sample, not current machine state`;
    const clipped=rows.filter(r=>finite(r.gradient_norm)&&r.gradient_norm>recipe.clip).length;
    $('clip-detail').textContent=`Clip threshold ${number(recipe.clip)} (dashed). Exceeded in ${clipped}/${rows.length} displayed updates. A large pre-clip norm is not automatically failure.`;
    const used=utilization(rows);$('padding').textContent=used===null?'Token/position counts unavailable.':`${number(100*used,1)}% valid positions · ${number(100*(1-used),1)}% padding in displayed updates. Rates exclude validation and checkpoint time.`;
    const summary=data.memory_summary;
    $('cuda-memory').textContent=`CUDA peak allocated: ${number(summary.cuda_peak_allocated_bytes/2**30,2)} GiB · reserved: ${number(summary.cuda_peak_reserved_bytes/2**30,2)} GiB. End-of-run peaks; not a live GPU meter.`;
    $('freshness').textContent=`Last metric write: ${number(data.update_age_seconds,0)} seconds ago. Last memory write: ${number(data.memory_age_seconds,0)} seconds ago. File freshness does not prove process health or successful exit.`;
    $('checkpoint-detail').textContent=data.checkpoints.length?'Saved files: '+data.checkpoints.map(c=>`${c.name} (${number(c.bytes/2**20,0)} MiB)`).join(' · '):'No completed checkpoint files found.';
    $('measurements').replaceChildren(...rows.slice(-200).map(r=>{
      const tr=document.createElement('tr');[r.update,r.loss,r.gradient_norm,r.valid_targets,r.seconds,r.lr].forEach((v,i)=>{const td=document.createElement('td');td.textContent=i===5&&finite(v)?v.toExponential(3):number(v,i===0||i===3?0:5);tr.append(td);});return tr;
    }));
    const prompts=[...new Set(data.snapshots.flatMap(s=>(s.samples||[]).map(p=>p.prompt)))];
    options($('prompt'),prompts.map(p=>({value:p,label:p})));
    const observations=data.snapshots.map(s=>({value:s.name,label:`Update ${s.update} · ${s.name.replace('.json','')}`}));
    options($('before'),observations,observations[0]?.value);options($('after'),observations,observations.at(-1)?.value);stories();
    const tokenAxis=$('axis').value==='tokens',xLabel=tokenAxis?'Valid training tokens':'Optimizer update';
    const x=r=>tokenAxis?r.cumulative_targets:r.update;
    const validationX=s=>!tokenAxis?s.update:s.update===0?0:rows.find(r=>r.update===s.update)?.cumulative_targets;
    charts=[
      ['loss-chart',[{label:'Training',points:seriesPoints(rows,x,r=>r.loss)},{label:'Validation',points:seriesPoints(data.snapshots,validationX,s=>s.validation?.loss)}],xLabel,'Mean NLL (nats)'],
      ['gradient-chart',[{label:'Gradient norm',points:seriesPoints(rows,x,r=>r.gradient_norm)}],xLabel,'L2 norm',recipe.clip],
      ['lr-chart',[{label:'Learning rate',points:seriesPoints(rows,x,r=>r.lr)}],xLabel,'Learning rate'],
      ['speed-chart',[{label:'Valid targets / second',points:seriesPoints(rows,x,r=>r.valid_targets_per_second)}],xLabel,'Tokens / second'],
      ['memory-chart',[{label:'Host available',points:seriesPoints(data.memory,r=>r.seconds,r=>r.available_gib)}],'Monitor elapsed seconds','Available GiB',data.reserve_gib],
      ['activation-chart',[{label:'RMS',points:seriesPoints((latest?.activations||[]).map((r,i)=>({...r,index:i})),r=>r.index,r=>r.rms)}],'Decoder stage','Activation RMS']
    ];draw();
  }
  async function refresh() {
    if(busy)return;busy=true;
    try{
      const response=await fetch('/api/runs',{cache:'no-store'});if(!response.ok)throw new Error('Cannot read run list');
      const runs=await response.json();const requested=new URLSearchParams(location.search).get('run');
      options($('run'),runs.map(r=>({value:r.name,label:r.name})),runs.some(r=>r.name===requested)?requested:runs[0]?.name);
      if(!runs.length){$('state').textContent='No run.json files found. Start a training run separately.';$('connection').textContent='Connected · no runs';$('content').hidden=true;return;}
      const name=$('run').value;
      const response2=await fetch('/api/run?name='+encodeURIComponent(name),{cache:'no-store'});if(!response2.ok)throw new Error('Run metadata unavailable or incomplete');
      const data=await response2.json();if(name!==$('run').value)return;render(data);
      $('connection').textContent='Connected · '+new Date().toLocaleTimeString();
    }catch(error){$('connection').textContent='Disconnected / unavailable — displayed data may be old';$('state').textContent=error.message;}
    finally{busy=false;}
  }
  $('run').addEventListener('change',()=>{state=null;$('before').replaceChildren();$('after').replaceChildren();refresh();});
  $('axis').addEventListener('change',()=>{if(state)render(state);});
  ['before','after','prompt','method'].forEach(id=>$(id).addEventListener('change',stories));
  new ResizeObserver(draw).observe(document.querySelector('main'));
  setInterval(()=>{if($('refresh').checked&&!document.hidden)refresh();},3000);
  refresh();
})();

// Optional inference is independent of chart refreshes and never changes model weights.
if (typeof document !== 'undefined') (() => {
  const element=id=>document.getElementById(id);
  const button=element('generate-button');
  async function modelInfo(){
    try {
      const response=await fetch('/api/playground');
      if(!response.ok)throw new Error('Model service unavailable');
      const info=await response.json();
      element('model-status').textContent=info.enabled ? `${info.run} · update ${info.update} · loaded on Spark` : 'Playground not enabled on this server';
      button.disabled=!info.enabled;
    } catch(error){element('model-status').textContent=error.message;}
  }
  element('generate-form').addEventListener('submit',async event=>{
    event.preventDefault();button.disabled=true;button.textContent='Generating…';
    const prompt=element('custom-prompt').value;
    element('generation-output').textContent='';element('generation-prompt').textContent=prompt;
    element('generation-status').textContent='Generating locally on Spark…';
    try {
      const response=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json','X-Playground-Request':'1'},body:JSON.stringify({prompt,temperature:Number(element('temperature').value),max_new_tokens:Number(element('generation-length').value),seed:Number(element('generation-seed').value)})});
      const result=await response.json();if(!response.ok)throw new Error(result.error||'Generation failed');
      element('generation-output').textContent=result.continuation||'(No visible continuation.)';
      const reasons={eos:'Model ended the story (EOS)',max_new_tokens:'Token limit reached',context_limit:'Context limit reached',time_limit:'60-second time limit reached',memory_reserve:'Stopped to preserve host memory'};
      element('generation-status').textContent=`${result.generated_tokens} tokens · ${result.seconds.toFixed(2)} seconds · ${reasons[result.stop_reason]||result.stop_reason} · seed ${result.seed}`;
    } catch(error){element('generation-status').textContent=error.message;}
    finally{button.disabled=false;button.textContent='Continue the story';}
  });
  modelInfo();
})();

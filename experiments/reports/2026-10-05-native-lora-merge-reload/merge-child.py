"""Fixed exact-base20-update LoRA merge/reload; local only, no training."""
from datetime import datetime,timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from dongxi_llms.run_identity import artifact_hashes,assert_compatible,collect_run_identity,tokenizer_interface
from run_chapter09_spark_sft import read_records
EVIDENCE=Path(__file__).parent
BASE=Path('/home/dongxi/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd')
ADAPTER=ROOT/'outputs/native-sft-lora-replay-20261005-original/policy'
OUTPUT=ROOT/'outputs/native-lora20-merged-20261005'
REVISION='da87bfb608c14b7cf20ba1ce41287e8de496c0cd'


def retain(name,value):
    with (EVIDENCE/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import PeftModel
    started=time.monotonic()
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    if shutil.disk_usage(ROOT/'outputs').free<4*1024**3:raise RuntimeError('4GiB output reserve required')
    genealogy=json.loads((ADAPTER/'course-genealogy.json').read_text())
    config=json.loads((ADAPTER/'adapter_config.json').read_text())
    if (genealogy['base_revision']!=REVISION or genealogy['base_model']!='Qwen/Qwen3-0.6B-Base'
        or config['r']!=8 or set(config['target_modules'])!={'q_proj','v_proj'}):
        raise ValueError('Unexpected exact-base/rank/target identity')
    base_files=artifact_hashes(BASE);adapter_files=artifact_hashes(ADAPTER)
    if genealogy['base_checkpoint_files']!=base_files:raise ValueError('Actual Base bytes differ from training genealogy')
    tokenizer=AutoTokenizer.from_pretrained(ADAPTER,local_files_only=True,trust_remote_code=False)
    interface=tokenizer_interface(tokenizer,template=tokenizer.chat_template,stop_ids=[151643,151645])
    assert_compatible(genealogy['checkpoint_interface'],interface,compare_source=False)
    identity=collect_run_identity(ROOT,source_files=[Path(__file__),ROOT/'src/dongxi_llms/run_identity.py'],
        input_files=[ROOT/'outputs/course-sft-interface-v1/dev.jsonl',ADAPTER/'course-genealogy.json'],
        checkpoint=BASE,environment_lock='/home/dongxi/dgx-spark-dongxi/uv.lock',
        config=dict(atol=.125,rtol=.015625,backend='sdpa',precision='BF16',adapter_files=adapter_files),
        interface=interface,device=dict(mode='cuda',name=torch.cuda.get_device_name()))
    retain('input-identity.json',identity)
    rows=read_records(ROOT/'outputs/course-sft-interface-v1/dev.jsonl')[:8]
    prefixes=[tokenizer.apply_chat_template(r['messages'][:-1],tokenize=True,
        add_generation_prompt=True,enable_thinking=False,return_dict=False) for r in rows]
    base=AutoModelForCausalLM.from_pretrained(BASE,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    model=PeftModel.from_pretrained(base,ADAPTER,local_files_only=True).eval()
    def logits(value):
        with torch.inference_mode():
            return [value(input_ids=torch.tensor([ids],device='cuda'),use_cache=False).logits.detach().float().cpu() for ids in prefixes]
    before=logits(model)
    merged=model.merge_and_unload(safe_merge=True).eval()
    after=logits(merged)
    comparisons=[]
    for row,a,b in zip(rows,before,after):
        torch.testing.assert_close(a,b,atol=.125,rtol=.015625)
        difference=a-b
        comparisons.append(dict(id=row['id'],shape=list(a.shape),max_absolute=float(difference.abs().max()),
            rms=float(difference.square().mean().sqrt()),last_position_argmax_equal=bool(a[:,-1].argmax()==b[:,-1].argmax())))
    OUTPUT.mkdir(mode=0o700)
    merged.save_pretrained(OUTPUT,safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT)
    retain('merged-interface.json',interface)
    with (OUTPUT/'course-genealogy.json').open('x') as f:
        json.dump(dict(kind='explicitly-merged-full-HF',base_model=genealogy['base_model'],base_revision=REVISION,
            base_checkpoint_files=base_files,adapter_files=adapter_files,checkpoint_interface=genealogy['checkpoint_interface'],
            scope='disposable20updateacceptance;notselected400updateparent'),f,indent=2);f.write('\n')
    del model,base,merged;gc.collect();torch.cuda.empty_cache()
    reloaded=AutoModelForCausalLM.from_pretrained(OUTPUT,local_files_only=True,trust_remote_code=False,
        dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    reload_logits=logits(reloaded)
    if not all(torch.equal(a,b) for a,b in zip(after,reload_logits)):
        raise ValueError('Actual merged reload logits differ')
    actual_tokenizer=AutoTokenizer.from_pretrained(OUTPUT,local_files_only=True,trust_remote_code=False)
    reconstructed=tokenizer_interface(actual_tokenizer,template=actual_tokenizer.chat_template,stop_ids=[151643,151645])
    assert_compatible(interface,reconstructed,compare_source=False)
    if artifact_hashes(BASE)!=base_files or artifact_hashes(ADAPTER)!=adapter_files:
        raise ValueError('Input model bytes changed during merge/reload')
    report=dict(status='passed',atol=.125,rtol=.015625,comparisons=comparisons,
        exact_reloaded_logits=True,interface_compatible=True,merged_files=artifact_hashes(OUTPUT),
        prompt_tokens=sum(map(len,prefixes)),forward_calls=24,
        full_prefix_positions=3*sum(map(len,prefixes)),
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-started,
        scope='sameSpark actualpretrainedBF16 merge/reload identity;notbitwiseadaptermerge/capability/generalhardwareclaim; costoutsideSFT/IObudgets')
    retain('verification.json',report);print(json.dumps(dict(status=report['status'],seconds=report['seconds'])))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as error:
        retain('failure.json',dict(type=type(error).__name__,message=str(error),scope='actualfailedmergeattempt;nocap/tolerancerelaxation'))
        raise

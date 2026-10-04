"""Bounded opt-in CUDA SFT runner for the Chapter 9 original instruction suite.

No execution on import. Revision must be an exact repository commit. See the
chapter lab for data generation, profile gates and evidence limits.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time
from uuid import uuid4

import torch
from torch.nn import functional as F

FAILURE_OUTPUT=None

def available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])/1024**2
    raise RuntimeError('Linux MemAvailable measurement unavailable')


def read_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not records or len({row['id'] for row in records}) != len(records):
        raise ValueError('Need nonempty unique-ID records')
    return records


def encode_record(tokenizer, record, max_length):
    messages = record['messages']
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('Training record needs a completed assistant answer')
    def render(value, generation=False):
        return tokenizer.apply_chat_template(value, tokenize=True, add_generation_prompt=generation,
                                             enable_thinking=False)
    ids = render(messages)
    if len(ids) > max_length:
        raise ValueError(f"Overlength record {record['id']}: reject, never silently truncate")
    labels = [-100]*len(ids)
    for index, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        prefix, through = render(messages[:index], True), render(messages[:index+1])
        # Refuse heuristic span labels if the tokenizer/template changes a prefix.
        if ids[:len(through)] != through or through[:len(prefix)] != prefix:
            raise ValueError('Template is not prefix-compatible; supply an audited template')
        labels[len(prefix):len(through)] = ids[len(prefix):len(through)]
    if not any(token != -100 for token in labels[1:]):
        raise ValueError('No surviving answer targets')
    return dict(id=record['id'], ids=ids, labels=labels)


def collate(records, pad_id, device):
    length=max(len(record['ids']) for record in records)
    ids=torch.full((len(records),length),pad_id,dtype=torch.long,device=device)
    labels=torch.full_like(ids,-100); attention=torch.zeros_like(ids)
    for row,record in enumerate(records):
        ids[row,:len(record['ids'])]=torch.tensor(record['ids'],device=device)
        labels[row,:len(record['ids'])]=torch.tensor(record['labels'],device=device)
        attention[row,:len(record['ids'])]=1
    return ids,labels,attention


def summed_loss(model,batch):
    ids,labels,attention=batch
    logits=model(input_ids=ids,attention_mask=attention,use_cache=False).logits[:,:-1].float()
    targets=labels[:,1:]
    return F.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1),
                           reduction='sum',ignore_index=-100)


def main():
    global FAILURE_OUTPUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',default='Qwen/Qwen3-0.6B-Base')
    parser.add_argument('--revision',required=True)
    parser.add_argument('--tokenizer-revision',required=True)
    parser.add_argument('--template',type=Path,required=True)
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--dev',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['full','lora'],default='full')
    parser.add_argument('--updates',type=int,default=20)
    parser.add_argument('--microbatch',type=int,default=1)
    parser.add_argument('--accumulation',type=int,default=4)
    parser.add_argument('--max-length',type=int,default=256)
    parser.add_argument('--learning-rate',type=float,default=2e-5)
    parser.add_argument('--runtime-seconds',type=int,default=900)
    parser.add_argument('--reserve-gib',type=float,default=25.)
    parser.add_argument('--seed',type=int,default=1212)
    parser.add_argument('--rank',type=int,default=8)
    parser.add_argument('--checkpoint-every',type=int,default=20)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    for revision in (args.revision,args.tokenizer_revision):
        if not re.fullmatch('[0-9a-f]{40}',revision):
            raise ValueError('Model and tokenizer revisions must be exact 40-character commit SHAs')
    if args.model not in ('Qwen/Qwen3-0.6B-Base','Qwen/Qwen3-1.7B-Base'):
        raise ValueError('Declared course run supports the two named base checkpoints')
    if min(args.updates,args.microbatch,args.accumulation,args.runtime_seconds,args.max_length,args.checkpoint_every)<=0:
        raise ValueError('Positive budgets required')
    if args.reserve_gib<25 or args.max_length>2048 or args.updates>2000:
        raise ValueError('Course bounds require >=25 GiB reserve, length<=2048, updates<=2000')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA with verified BF16 support required')
    if available_gib()<args.reserve_gib:
        raise RuntimeError('Host memory reserve unavailable before load')
    start=time.monotonic(); minimum_available=available_gib(); total_targets=0
    invocation_id=uuid4().hex
    def guard():
        nonlocal minimum_available
        current=available_gib(); minimum_available=min(minimum_available,current)
        if current<args.reserve_gib: raise RuntimeError('Host memory reserve violated')
        if time.monotonic()-start>args.runtime_seconds: raise TimeoutError('Declared runtime cap reached')
    output=args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError('Use a new empty output directory or explicit resume')
    output.mkdir(parents=True,exist_ok=True)
    FAILURE_OUTPUT=output/'failure.json'
    train_records,dev_records=read_records(args.train),read_records(args.dev)
    if {record['group'] for record in train_records}&{record['group'] for record in dev_records}:
        raise ValueError('Train/development source groups overlap')
    from transformers import AutoTokenizer,AutoModelForCausalLM
    import transformers
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    guard()
    tokenizer=AutoTokenizer.from_pretrained(args.model,revision=args.tokenizer_revision)
    guard()
    tokenizer.chat_template=args.template.read_text()
    for marker in ('<|im_start|>','<|im_end|>'):
        encoded=tokenizer.encode(marker,add_special_tokens=False)
        if len(encoded)!=1 or encoded[0]==tokenizer.unk_token_id:
            raise ValueError('Template requires existing single-token message markers; adding vocabulary is out of scope')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    train=[encode_record(tokenizer,record,args.max_length) for record in train_records]
    dev=[encode_record(tokenizer,record,args.max_length) for record in dev_records]
    end_message_id=tokenizer.convert_tokens_to_ids('<|im_end|>')
    stop_ids=sorted({value for value in (end_message_id,tokenizer.eos_token_id) if value is not None})
    config={key:str(value) if isinstance(value,Path) else value for key,value in vars(args).items() if key not in ('resume','output')}
    config.update(torch_version=torch.__version__,transformers_version=transformers.__version__,
                  template_sha256=hashlib.sha256(args.template.read_bytes()).hexdigest(),
                  train_sha256=hashlib.sha256(args.train.read_bytes()).hexdigest(),
                  dev_sha256=hashlib.sha256(args.dev.read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(),dtype='BF16 weights/autocast; FP32 cross entropy',
                  attention_backend='sdpa',loss_policy='assistant body + template end tokens; one explicit shift')
    config['generation_stop_ids']=stop_ids
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    model=AutoModelForCausalLM.from_pretrained(args.model,revision=args.revision,
                                             torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda()
    guard()
    model.config.use_cache=False
    model.gradient_checkpointing_enable()
    if args.mode=='lora':
        from peft import LoraConfig,get_peft_model
        model=get_peft_model(model,LoraConfig(r=args.rank,lora_alpha=args.rank,lora_dropout=0.,
                                             target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM',revision=args.revision))
        model.enable_input_require_grads()
    parameters=[parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,weight_decay=0.)
    guard()
    order=list(range(len(train))); random.Random(args.seed).shuffle(order)
    cursor,first_update=0,0
    if args.resume:
        saved=torch.load(args.resume,map_location='cpu',weights_only=False)
        if saved['config']!=config:
            raise ValueError('Resume configuration/data/template identity mismatch')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        cursor,first_update=saved['cursor'],saved['update']
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    guard()
    def evaluate():
        model.eval(); loss_sum,count=0.,0
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for record in dev:
                guard(); batch=collate([record],tokenizer.pad_token_id,'cuda')
                loss_sum+=float(summed_loss(model,batch)); count+=int((batch[1][:,1:]!=-100).sum())
        model.train()
        return loss_sum/count
    def generate_samples():
        model.eval(); samples=[]
        with torch.no_grad():
            for record in dev_records[:8]:
                guard()
                prefix=tokenizer.apply_chat_template(record['messages'][:-1],tokenize=True,
                                                     add_generation_prompt=True,enable_thinking=False,return_tensors='pt').cuda()
                if prefix.shape[1]+64>args.max_length:
                    raise ValueError('Fixed 64-token evaluation plus prompt exceeds declared context; increase context explicitly')
                generated=model.generate(prefix,attention_mask=torch.ones_like(prefix),max_new_tokens=64,
                                         do_sample=False,pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=stop_ids,use_cache=True)
                continuation=generated[0,prefix.shape[1]:].tolist()
                answer=tokenizer.decode(continuation,skip_special_tokens=True).strip()
                samples.append(dict(id=record['id'],generated_ids=continuation,
                                    text=tokenizer.decode(continuation,skip_special_tokens=False),
                                    reference=record['messages'][-1]['content'],
                                    exact_match=answer==record['messages'][-1]['content'],
                                    stopped_on_end=bool(continuation and continuation[-1]==end_message_id),
                                    final_stop_id=continuation[-1] if continuation and continuation[-1] in stop_ids else None))
        model.train()
        return samples
    initial=evaluate(); baseline_samples=generate_samples()
    def save_checkpoint(update):
        guard()
        pending=output/'checkpoint-next.pt'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=config,
                        cursor=cursor,update=update,torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all()),pending)
        pending.replace(output/'checkpoint.pt')
        guard()
    with (output/'metrics.jsonl').open('a') as log:
        for update in range(first_update,args.updates):
            guard(); window=[]
            for _ in range(args.accumulation):
                records=[train[order[(cursor+i)%len(order)]] for i in range(args.microbatch)]
                cursor+=args.microbatch; window.append(collate(records,tokenizer.pad_token_id,'cuda'))
            count=sum(int((batch[1][:,1:]!=-100).sum()) for batch in window)
            optimizer.zero_grad(set_to_none=True); recorded_sum=0.
            for batch in window:
                guard()
                with torch.autocast('cuda',dtype=torch.bfloat16): loss=summed_loss(model,batch)
                if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
                (loss/count).backward(); recorded_sum+=float(loss.detach())
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm): raise RuntimeError('Nonfinite accumulated gradient')
            optimizer.step(); total_targets+=count
            log.write(json.dumps(dict(invocation_id=invocation_id,resume_checkpoint_update=first_update,
                                      update=update+1,answer_nll=recorded_sum/count,targets=count,
                                      gradient_norm=float(norm),elapsed_seconds=time.monotonic()-start))+'\n'); log.flush()
            if (update+1)%args.checkpoint_every==0:
                save_checkpoint(update+1)
    final=evaluate(); samples=generate_samples()
    save_checkpoint(args.updates)
    tokenizer.save_pretrained(output/'tokenizer')
    model.save_pretrained(output/'policy',safe_serialization=True)
    tokenizer.save_pretrained(output/'policy')
    (output/'policy'/'course-genealogy.json').write_text(json.dumps(
        dict(kind='full-HF-model' if args.mode=='full' else 'PEFT-adapter-requires-pinned-base',
             base_model=args.model,base_revision=args.revision,mode=args.mode,
             template_sha256=config['template_sha256'],data_sha256=config['train_sha256']),indent=2)+'\n')
    result=dict(status='completed',invocation_id=invocation_id,resume_checkpoint_update=first_update,
                updates=args.updates,initial_dev_nll=initial,final_dev_nll=final,
                supervised_targets_this_invocation=total_targets,elapsed_seconds=time.monotonic()-start,
                minimum_sampled_memavailable_gib=minimum_available,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                trainable_parameters=sum(p.numel() for p in parameters),
                baseline_samples=baseline_samples,samples=samples,
                generation_scope='first eight development IDs, deterministic greedy64; not the frozen publication suite')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        if FAILURE_OUTPUT is not None:
            FAILURE_OUTPUT.write_text(json.dumps(dict(status='failed',exception_type=type(exc).__name__,
                                                     evidence='partial logs retained; no completion claim'),indent=2)+'\n')
        raise

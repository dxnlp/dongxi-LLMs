// Draft-only transfer adapter: X converts GIF placeholders into video blocks.
// All browser interactions use the supported cua_repl Tab handle passed in.
import * as xh from '/Users/yongchao/.codex/skills/x-article-drafter/scripts/x_inapp_article_helpers.mjs';

export async function state(tab) {
  return tab.playwright.evaluate(() => {
    const editor=document.querySelector('.public-DraftEditor-content[contenteditable="true"]');
    if(!editor) throw Error('Missing editor');
    const norm=s=>(s||'').replace(/\s+/g,' ').trim();
    const blocks=Array.from(editor.querySelectorAll('[data-block="true"]'))
      .filter(b=>!b.parentElement.closest('[data-block="true"]')).map((b,i)=>({
      i,text:norm(b.innerText||b.textContent),
      media:!!b.querySelector('img,video,[aria-label="Remove media"]'),
      video:!!b.querySelector('video'),
    }));
    const textBlocks=blocks.filter(b=>!b.media&&b.text);
    const mediaBlocks=blocks.filter(b=>b.media);
    const pageText=document.body.innerText;
    const cover=Array.from(document.querySelectorAll('img')).filter(im=>!editor.contains(im)&&/pbs\.twimg\.com\/media|blob:https:\/\/x\.com/.test(im.currentSrc||im.src)).length;
    return {url:location.href,title:document.querySelector('textarea')?.value,
      blocks,textBlocks:textBlocks.map(b=>({...b})),mediaBlocks:mediaBlocks.map(b=>({...b})),text:textBlocks.map(b=>b.text).join(' '),cover,
      wordCount:pageText.match(/\d[\d,]* words/)?.[0]||null,
      lastSaved:pageText.match(/Last saved[^\n]*(?:\n[^\n]*)?/)?.[0]||null,
      videos:Array.from(editor.querySelectorAll('video')).map(v=>({src:v.currentSrc||v.src,
        duration:v.duration,time:v.currentTime,paused:v.paused,readyState:v.readyState,loop:v.loop})),
      h2:editor.querySelectorAll('h2').length};
  });
}

export async function audit(tab,planPath,draftUrl) {
  const p=await xh.buildSequentialBodySegments(planPath),s=await state(tab);
  if(s.url!==draftUrl)throw Error('Wrong draft');
  const slots=s.mediaBlocks.map((b,i)=>{
    const prev=s.textBlocks.filter(t=>t.i<b.i).at(-1)?.text||'';
    const next=s.textBlocks.find(t=>t.i>b.i)?.text||'';
    const target=p.images[i];
    return {no:i+1,previousOk:!!target&&prev===xh.normalizeText(target.target_after_text_full||target.target_after_text),
      nextOk:!!target&&next===xh.normalizeText(target.next_text_full||target.next_text)};
  });
  const sourceTextMatches=s.text===p.source_text;
  const allSlotsOk=slots.length===p.images.length&&slots.every(v=>v.previousOk&&v.nextOk);
  const captionsMatch=s.mediaBlocks.length===p.images.length&&s.mediaBlocks.every((b,i)=>b.text===xh.normalizeText(p.images[i].caption));
  return {complete:s.title===p.plan.title&&sourceTextMatches&&allSlotsOk&&captionsMatch&&s.cover===1&&s.textBlocks.length===p.plan.total_blocks,
    url:s.url,title:s.title,wordCount:s.wordCount,lastSaved:s.lastSaved,
    sourceTextMatches,textBlocks:s.textBlocks.length,expectedTextBlocks:p.plan.total_blocks,
    inlineMedia:s.mediaBlocks.length,coverMedia:s.cover,totalMedia:s.cover+s.mediaBlocks.length,
    placeholders:(s.text.match(/\[IMAGE \d+\]/g)||[]).length,h2:s.h2,allSlotsOk,captionsMatch,slots,videos:s.videos,
    publishClicked:false};
}

export async function appendPair(tab,planPath,draftUrl,index) {
  const p=await xh.buildSequentialBodySegments(planPath);
  let s=await state(tab);
  if(s.url!==draftUrl||s.cover!==1||s.title!==p.plan.title)throw Error('Draft preflight mismatch');
  const previous=index?p.segments[index-1].cumulative_text:'';
  if(s.text!==previous||s.mediaBlocks.length!==index)throw Error('Resume boundary mismatch');
  const last=s.blocks.at(-1);
  if(last?.text||last?.media)throw Error('Expected trailing empty block');
  const segment=p.segments[index];
  await tab.playwright.locator('.public-DraftEditor-content[contenteditable="true"] [data-block="true"]').last().click();
  await tab.paste(segment.html,{format:'html'});
  await tab.getAXState({emit:false});
  s=await state(tab);
  if(s.text!==segment.cumulative_text||s.mediaBlocks.length!==index)throw Error('HTML mismatch; stopped without retry');
  const image=p.images[index];
  if(image){
    await xh.copyImageToTabClipboard(tab,image.path);
    await tab.pressKey('super+v');
    await tab.getAXState({emit:false});
    s=await state(tab);
    // Media processing outlives a single browser call. Never repaste; caller
    // checks state in a later call before advancing to the next guarded pair.
    if(s.text===segment.cumulative_text&&s.mediaBlocks.length===index)
      return {segment:index,pendingMedia:index+1,action:'Inspect state later; do not repaste'};
    if(s.text!==segment.cumulative_text||s.mediaBlocks.length!==index+1)throw Error('Media mismatch; inspect before resume');
    const target=xh.normalizeText(image.target_after_text_full||image.target_after_text);
    const lastText=s.textBlocks.at(-1);
    if(lastText.text!==target||s.mediaBlocks.at(-1).i<=lastText.i)throw Error('Media placement mismatch');
  }
  return {segment:index,textBlocks:s.textBlocks.length,inlineMedia:s.mediaBlocks.length,lastSaved:s.lastSaved};
}

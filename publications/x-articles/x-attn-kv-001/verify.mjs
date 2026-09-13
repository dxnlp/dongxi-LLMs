/** Audit bilingual local packages, image slots, arithmetic assets and players. */
import {readFile,writeFile,mkdir,access} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
import {chromium} from '../../../visuals/animations/projects/looped-transformer/node_modules/playwright/index.mjs';
const p=path.dirname(fileURLToPath(import.meta.url));
const m=JSON.parse(await readFile(path.join(p,'metadata.json')));
for(const e of m.files){
  const hash=createHash('sha256').update(await readFile(path.join(p,e.path))).digest('hex');
  if(hash!==e.sha256)throw Error('Asset hash mismatch: '+e.path);
  const name=path.basename(e.path);
  const loopAsset=Object.keys(m.animations).some(stem=>name===stem+'.gif'||name===stem+'-still.png');
  const expected=(name==='cover.png'||name===path.basename(m.cover.path))?[2000,800]:(loopAsset?[960,540]:(m.equations[name]?.size??[1600,900]));
  if(JSON.stringify(e.dimensions)!==JSON.stringify(expected))throw Error('Wrong dimensions');
}
const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
const results=[];const errors=[];
await mkdir(path.join(p,'qa'),{recursive:true});
try{
  for(const lang of ['en','zh']){
    const dir=lang==='en'?p:path.join(p,'zh');
    const plan=JSON.parse(await readFile(path.join(dir,'x-editor-clean-body-image-plan.json')));
    const clean=await readFile(path.join(dir,'x-editor-clean-body.html'),'utf8');
    const source=await readFile(path.join(dir,'x-editor-clean-body.md'),'utf8');
    if(plan.image_count!==7||plan.missing_image_count!==0)throw Error('Incomplete image plan');
    if(JSON.stringify(plan.images.map(im=>path.basename(im.path)))!==JSON.stringify(m.inline_image_order))throw Error('Incorrect equation/diagram order');
    if(/\[IMAGE\s+\d+\]|<h1>|<video|<img/.test(clean))throw Error('Transfer body contains title, placeholders or media');
    if(/<a\b|https?:\/\//i.test(clean)||/https?:\/\//i.test(source))throw Error('Article links remain');
    if(/<pre\b|def(?:&nbsp;|\s)+attend_new/.test(clean))throw Error('Removed code example remains');
    if(lang==='zh'&&/不是|而是|并非|契约|接口|借口/.test(source))throw Error('Chinese editorial rule failed');
    const manuscript=await readFile(path.join(dir,'x-editor-draft-body-with-image-placeholders.md'),'utf8');
    if(/一个小实验|A small experiment|1\.80691|40\/12|math-projections|03-cache-evidence/.test(manuscript+clean))throw Error('Removed experiment content remains');
    const prose=manuscript.replace(/```[\s\S]*?```/g,'').replace(/`[^`]+`/g,'');
    if(/output_|q_t|K_≤|V_≤|H_KV|d_model|bytes_per_element|\$\$|\\frac/.test(prose))throw Error('Unrendered mathematical source remains');
    for(const image of plan.images){
      await access(image.path);
      if(!image.target_after_text_full||!image.next_text_full)throw Error('Missing image anchors');
      if(clean.split(image.target_block_html).length!==2)throw Error('Ambiguous image anchor');
    }
    const page=await browser.newPage({viewport:{width:1440,height:1100}});
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(pathToFileURL(path.join(dir,'review.html')).href);
    const imgs=await page.locator('img').evaluateAll(async images=>{await Promise.all(images.map(im=>im.decode()));return images.map(im=>({width:im.naturalWidth,height:im.naturalHeight}));});
    if(imgs.length!==8||imgs.some(im=>!im.width))throw Error('Missing review images');
    if(!await page.locator('main > img').evaluate((im,cover)=>im.src.endsWith(cover)&&im.naturalWidth===2000&&im.naturalHeight===800,m.cover.path))throw Error('Incorrect cover selection or aspect ratio');
    const links=await page.locator('a').evaluateAll(as=>as.map(a=>a.href));
    if(await page.locator('article a').count())throw Error('Review article links remain');
    if(await page.locator('article pre').count())throw Error('Removed code example remains');
    if(await page.locator('.animation').count()!==5||plan.images.filter(i=>i.path.endsWith('.gif')).length!==5)throw Error('Missing GIF or upload slot');
    for(const figure of await page.locator('.animation').all()){
      const anim=figure.locator('img'),toggle=figure.locator('button');
      const shot1=await anim.screenshot();let advanced=false;
      for(let attempt=0;attempt<4&&!advanced;attempt++){
        await page.waitForTimeout(700);advanced=!shot1.equals(await anim.screenshot());
      }
      if(!advanced)throw Error('GIF did not visibly advance');
      await toggle.click();
      if(!await anim.evaluate(im=>im.src===new URL(im.dataset.still,document.baseURI).href))throw Error('Pause fallback failed');
      const paused1=await anim.screenshot();await page.waitForTimeout(250);
      if(!paused1.equals(await anim.screenshot()))throw Error('Paused image changed');
      await toggle.click();
      if(!await anim.evaluate(im=>im.src===new URL(im.dataset.motion,document.baseURI).href))throw Error('Resume failed');
    }
    for(const href of links.filter(h=>h.startsWith('file:')))await access(fileURLToPath(href));
    await page.screenshot({path:path.join(p,'qa',lang+'-desktop.png'),fullPage:false});
    if(await page.locator('.equation').count()!==2)throw Error('Missing typeset equations');
    for(const [i,eq] of (await page.locator('.equation').all()).entries()){
      await eq.screenshot({path:path.join(p,'qa',lang+'-math-'+(i+1)+'-desktop.png')});
    }
    if(lang==='en')await page.locator('details').evaluate(el=>el.open=true);
    else if(await page.locator('details, video').count())throw Error('Removed Chinese companion section remains');
    const media=[];
    for(const v of await page.locator('video').all()){
      await v.evaluate(async el=>{el.muted=true;await el.play();});
      await page.waitForFunction(el=>el.currentTime>0.15,await v.elementHandle(),{timeout:15000});
      media.push(await v.evaluate(el=>{el.pause();return {width:el.videoWidth,height:el.videoHeight,duration:el.duration};}));
    }
    await page.setViewportSize({width:390,height:844});
    await page.evaluate(()=>scrollTo(0,0));
    if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');
    await page.screenshot({path:path.join(p,'qa',lang+'-mobile.png')});
    for(const [i,eq] of (await page.locator('.equation').all()).entries()){
      await eq.screenshot({path:path.join(p,'qa',lang+'-math-'+(i+1)+'-mobile.png')});
    }
    if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile math overflow');
    for(const [i,figure] of (await page.locator('.animation').all()).entries()){
      await figure.locator('button').click();
      await figure.screenshot({path:path.join(p,'qa',lang+'-animation-'+(i+1)+'-mobile.png')});
    }
    await page.emulateMedia({reducedMotion:'reduce'});
    await page.reload();
    if(!await page.locator('.animation img').evaluateAll(images=>images.every(im=>im.currentSrc===new URL(im.dataset.still,document.baseURI).href)))throw Error('Reduced-motion fallback failed');
    if(!await page.locator('.animation-toggle').evaluateAll(buttons=>buttons.every(b=>b.getAttribute('aria-pressed')==='false')))throw Error('Reduced-motion button state failed');
    results.push({language:lang,word_count:source.trim().split(/\s+/).length,cjk_character_count:(source.match(/[\u3400-\u9fff]/g)||[]).length,
      review_images:imgs.length,inline_images:plan.image_count,animated_gifs:5,gif_playback:true,pause_resume:true,reduced_motion_fallback:true,typeset_equations:2,removed_experiment_absent:true,unrendered_math:0,placeholders:0,article_links:0,inline_code_examples:0,cover_ratio:'5:2',local_links_resolve:true,mobile_overflow:false,companion_media:media});
    await page.close();
  }
  if(errors.length)throw Error(errors.join('\n'));
  const report={status:'passed',scope:'Local package only; no X editor interaction',published:false,transferred:false,asset_hashes:true,browser_errors:errors,languages:results};
  await writeFile(path.join(p,'qa.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
}catch(error){
  await writeFile(path.join(p,'qa.json'),JSON.stringify({status:'failed',scope:'Local package only',error:String(error),published:false,transferred:false},null,2)+'\n');
  throw error;
}finally{await browser.close();}

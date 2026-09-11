/** Check reproducible sources/media plus both selectors and individual players. */
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
import {chromium} from '../looped-transformer/node_modules/playwright/index.mjs';
const p=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(p,'../../../..');
const m=JSON.parse(await readFile(path.join(p,'metadata.json')));
if(m.preview)throw Error('Full-resolution render required');
for(const entry of [...m.sources,...m.outputs]){
  const hash=createHash('sha256').update(await readFile(path.join(root,entry.path))).digest('hex');
  if(hash!==entry.sha256)throw Error(`Hash mismatch: ${entry.path}`);
}
const errors=[],clips=[];let browser;
async function checkPoster(page){
  const poster=await page.locator('video').evaluate(async v=>{
    const image=new Image();image.src=v.poster;await image.decode();
    return {url:v.poster,width:image.naturalWidth,height:image.naturalHeight};
  });
  if(poster.url!==pathToFileURL(path.join(p,'assets/deepseek-v41-architecture-poster.png')).href||poster.width!==1920||poster.height!==1080)throw Error('Wrong architecture poster');
  return poster;
}
try{
  browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(p,'review.html')).href);
  for(let i=0;i<m.clips.length;i++){
    await page.locator(`button[data-index="${i}"]`).click();
    await page.waitForFunction(()=>document.querySelector('video').readyState>=1);
    const poster=await checkPoster(page);
    await page.locator('video').screenshot({path:path.join(p,'rendered',m.clips[i].id,'browser-poster.png')});
    if(/\bCED\b|Excalidraw|Manim|ANIM-|本地审阅版|[\u4e00-\u9fff]/.test(await page.locator('body').innerText()))throw Error('Abbreviation, production labels or non-English copy');
    await page.locator('video').evaluate(async v=>{v.muted=true;await v.play()});
    await page.waitForFunction(()=>document.querySelector('video').currentTime>.25,{},{timeout:15000});
    const media=await page.locator('video').evaluate(v=>{v.pause();return {width:v.videoWidth,height:v.videoHeight,duration:v.duration}});
    if(media.width!==1920||media.height!==1080)throw Error('Wrong resolution');
    await page.locator('video').evaluate(v=>new Promise(resolve=>{v.addEventListener('seeked',resolve,{once:true});v.currentTime=v.duration-1;}));
    await page.screenshot({path:path.join(p,'rendered',m.clips[i].id,'browser-review.png'),fullPage:true});
    clips.push({id:m.clips[i].id,playback:true,poster,...media});
  }
  await page.setViewportSize({width:390,height:844});
  if(!await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth))throw Error('Mobile overflow');
  for(const c of m.clips){
    await page.goto(pathToFileURL(path.join(p,c.id+'.html')).href);
    await page.waitForFunction(()=>document.querySelector('video').readyState>=1);
    if(await page.locator('h1').innerText()!==c.title)throw Error('Standalone mismatch');
    await checkPoster(page);
  }
  if(errors.length)throw Error(errors.join('\n'));
  const qa={hashes_match:true,both_selectors:true,individual_players:true,shared_architecture_poster:true,mobile_overflow:false,browser_errors:errors,clips};
  await writeFile(path.join(p,'qa.json'),JSON.stringify(qa,null,2)+'\n');console.log(JSON.stringify(qa,null,2));
}finally{if(browser)await browser.close()}

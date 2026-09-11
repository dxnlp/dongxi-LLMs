/** Exercise every series selector and individual player, then check manifests. */
import {readFile,writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
import {chromium} from '../node_modules/playwright/index.mjs';
const p=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(p,'../../../../..');
const metadata=JSON.parse(await readFile(path.join(p,'metadata.json')));
for(const entry of [...metadata.sources,...metadata.outputs]){
  const hash=createHash('sha256').update(await readFile(path.join(root,entry.path))).digest('hex');
  if(hash!==entry.sha256)throw Error(`Manifest mismatch: ${entry.path}`);
}
const errors=[];const results=[];let browser;
try{
  browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page=await browser.newPage({viewport:{width:1440,height:1100}});
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(p,'review.html')).href);
  for(let i=0;i<metadata.clips.length;i++){
    await page.locator(`button[data-index="${i}"]`).click();
    await page.waitForFunction(()=>document.querySelector('video').readyState>=1);
    const text=await page.locator('body').innerText();
    if(/Excalidraw|Manim|ANIM-|本地审阅版|[\u4e00-\u9fff]/.test(text))throw Error('Unexpected learner-facing labels');
    await page.locator('video').evaluate(async v=>{v.muted=true;await v.play()});
    await page.waitForFunction(()=>document.querySelector('video').currentTime>.25,{},{timeout:15000});
    const media=await page.locator('video').evaluate(v=>{v.pause();return {width:v.videoWidth,height:v.videoHeight,duration:v.duration}});
    if(media.width!==1920||media.height!==1080)throw Error('Expected full-resolution video');
    await page.locator('video').evaluate(v=>new Promise(resolve=>{v.addEventListener('seeked',resolve,{once:true});v.currentTime=v.duration-1;}));
    await page.screenshot({path:path.join(p,'rendered',metadata.clips[i].id,'browser-review.png'),fullPage:true});
    results.push({id:metadata.clips[i].id,playback:true,...media});
  }
  await page.setViewportSize({width:390,height:844});
  if(!(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)))throw Error('Mobile overflow');
  for(const c of metadata.clips){
    await page.goto(pathToFileURL(path.join(p,c.id+'.html')).href);
    await page.waitForFunction(()=>document.querySelector('video').readyState>=1);
    if(await page.locator('h1').innerText()!==c.title)throw Error('Wrong individual player');
  }
  if(errors.length)throw Error(errors.join('\n'));
  const qa={manifest_hashes_match:true,all_three_selectors:true,all_individual_players:true,
    english_only:true,production_labels_absent:true,mobile_overflow:false,browser_errors:errors,clips:results};
  await writeFile(path.join(p,'qa.json'),JSON.stringify(qa,null,2)+'\n');
  console.log(JSON.stringify(qa,null,2));
}finally{if(browser)await browser.close()}

/** Check deterministic exports, media integrity, and actual browser playback. */
import {readFile,writeFile,readdir} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath,pathToFileURL} from 'node:url';
import path from 'node:path';
import {chromium} from 'playwright';
const project=path.dirname(fileURLToPath(import.meta.url));
const root=path.resolve(project,'../../../..');
const hash=async p=>createHash('sha256').update(await readFile(p)).digest('hex');
const assets=(await readdir(path.join(project,'assets'))).map(n=>path.join(project,'assets',n));
const before=await Promise.all(assets.map(hash));
execFileSync(process.execPath,[path.join(project,'export.mjs')],{stdio:'inherit'});
const after=await Promise.all(assets.map(hash));
if(JSON.stringify(before)!==JSON.stringify(after))throw Error('Nondeterministic export');
for(const p of assets.filter(p=>p.endsWith('.svg'))){if(/<text[\s>]/.test(await readFile(p,'utf8')))throw Error('SVG text');}
const metadata=JSON.parse(await readFile(path.join(project,'metadata.json')));
for(const entry of [...metadata.sources,...metadata.outputs]){
  if(await hash(path.join(root,entry.path))!==entry.sha256)throw Error(`Hash mismatch ${entry.path}`);
}
let browser;const errors=[];
try{
  browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page=await browser.newPage({viewport:{width:1440,height:1080},deviceScaleFactor:1});
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto(pathToFileURL(path.join(project,'review.html')).href);
  const text=await page.locator('body').innerText();
  if(/Excalidraw|Manim|ANIM-|本地审阅版/.test(text))throw Error('Production text leaked into learner page');
  await page.locator('video').evaluate(v=>new Promise((resolve,reject)=>{
    if(v.readyState>=1)resolve();else{v.onloadedmetadata=resolve;v.onerror=()=>reject(Error('Media error'));}
  }));
  const media=await page.locator('video').evaluate(v=>({width:v.videoWidth,height:v.videoHeight,duration:v.duration}));
  if(media.width!==1920||media.height!==1080)throw Error('Expected final 1080p video');
  await page.locator('video').evaluate(async v=>{v.muted=true;await v.play()});
  await page.waitForFunction(()=>document.querySelector('video').currentTime>.25,{},{timeout:15000});
  const advanced=await page.locator('video').evaluate(v=>{v.pause();return v.currentTime>.1});
  if(!advanced)throw Error('Playback did not advance');
  await page.locator('video').evaluate(v=>new Promise(resolve=>{v.addEventListener('seeked',resolve,{once:true});v.currentTime=v.duration-1;}));
  await page.screenshot({path:path.join(project,'rendered/browser-review.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  const noOverflow=await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth);
  if(!noOverflow||errors.length)throw Error('Responsive page or browser error');
  const qa={deterministic_export:true,asset_count:assets.length,svg_text:false,
    manifest_hashes_match:true,production_labels_absent:true,browser_playback:true,
    mobile_overflow:false,browser_errors:errors,media};
  await writeFile(path.join(project,'qa.json'),JSON.stringify(qa,null,2)+'\n');
  console.log(JSON.stringify(qa,null,2));
}finally{if(browser)await browser.close()}

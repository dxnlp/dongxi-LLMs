/** Export seeded Excalidraw paths. Native Manim text is added in the scene. */
import {build} from 'esbuild';
import {chromium} from 'playwright';
import {readFile,writeFile,mkdir} from 'node:fs/promises';
import {createServer} from 'node:http';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const project=path.dirname(fileURLToPath(import.meta.url));
await mkdir(path.join(project,'assets'),{recursive:true});
await mkdir(path.join(project,'.export-build'),{recursive:true});
await build({stdin:{contents:'import {convertToExcalidrawElements,exportToSvg} from "@excalidraw/excalidraw"; window.bridge={convertToExcalidrawElements,exportToSvg};',resolveDir:project},bundle:true,format:'iife',outfile:path.join(project,'.export-build/bridge.js'),define:{'process.env.NODE_ENV':'"production"'},loader:{'.woff2':'empty','.css':'empty'}});
const script=await readFile(path.join(project,'.export-build/bridge.js'));
const server=createServer((req,res)=>{
  if(req.url==='/bridge.js'){res.setHeader('Content-Type','application/javascript');res.end(script)}
  else{res.setHeader('Content-Type','text/html');res.end('<!doctype html><script src="/bridge.js"></script>')}
});
await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
let browser;
try{
  browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page=await browser.newPage();
  await page.goto(`http://127.0.0.1:${server.address().port}`);
  await page.waitForFunction(()=>Boolean(window.bridge));
  const sketches=JSON.parse(await readFile(path.join(project,'sketches.json'),'utf8'));
  for(const [name,input] of Object.entries(sketches)){
    const result=await page.evaluate(async input=>{
      const elements=window.bridge.convertToExcalidrawElements(input,{regenerateIds:false});
      for(const e of elements){e.updated=0;e.versionNonce=0}
      const appState={viewBackgroundColor:'#ffffff',exportBackground:false,exportWithDarkMode:false};
      const svg=await window.bridge.exportToSvg({elements:elements.filter(e=>e.type!=='text'),appState,files:{},exportPadding:0});
      return {svg:svg.outerHTML,scene:{type:'excalidraw',version:2,source:'Dongxi ANIM-LOOP-001',elements,appState,files:{}}};
    },input);
    if(/<text[\s>]/.test(result.svg)) throw Error('Unexpected SVG text');
    await writeFile(path.join(project,'assets',`${name}.svg`),result.svg+'\n');
    await writeFile(path.join(project,'assets',`${name}.excalidraw`),JSON.stringify(result.scene,null,2)+'\n');
    console.log(`Exported ${name}`);
  }
}finally{if(browser)await browser.close();await new Promise(resolve=>server.close(resolve))}

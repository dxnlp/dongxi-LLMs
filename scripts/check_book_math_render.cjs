// Optional local TeX-to-SVG check. See docs/MATH_FORMATTING.md for setup.
// This does not emulate GitHub's Markdown parser or macro sanitizer.
const {execFileSync} = require('node:child_process');
const path = require('node:path');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const tex = new TeX({
  packages: ['base', 'ams'],
  formatError(_jax, error) { throw error; },
});
const mathDocument = mathjax.document('', {
  InputJax: tex, OutputJax: new SVG({fontCache: 'none'}),
});
const files = JSON.parse(execFileSync('python3', [
  path.join(__dirname, 'check_book_math.py'), '--json',
], {encoding: 'utf8'}));
let passed = 0, failed = 0;
for (const file of files) {
  for (const expression of file.expressions) {
    try {
      tex.reset();
      const svg = adaptor.outerHTML(mathDocument.convert(expression.tex, {
        display: expression.display,
      }));
      if (!svg.includes('<svg') || svg.includes('data-mjx-error') || svg.includes('<merror')) {
        throw Error('Renderer returned an error or no SVG');
      }
      passed++;
    } catch (error) {
      failed++;
      console.error(`${file.path}:${expression.line}: ${error}`);
    }
  }
}
console.log(JSON.stringify({renderer: 'MathJax', passed, failed}));
process.exitCode = failed ? 1 : 0;

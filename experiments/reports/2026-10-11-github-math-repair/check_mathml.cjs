// Compare original equations, repaired source and GitHub HTML after the
// observed client's document.write decoding. DOM emulation is not browser layout.
const fs = require('node:fs');
const path = require('node:path');
const {execFileSync} = require('node:child_process');
const crypto = require('node:crypto');
const {JSDOM} = require('jsdom');
const {mathjax} = require('mathjax-full/js/mathjax.js');
const {TeX} = require('mathjax-full/js/input/tex.js');
const {SVG} = require('mathjax-full/js/output/svg.js');
const {liteAdaptor} = require('mathjax-full/js/adaptors/liteAdaptor.js');
const {RegisterHTMLHandler} = require('mathjax-full/js/handlers/html.js');
const {STATE} = require('mathjax-full/js/core/MathItem.js');
const {SerializedMmlVisitor} = require('mathjax-full/js/core/MmlTree/SerializedMmlVisitor.js');
require('mathjax-full/js/input/tex/ams/AmsConfiguration.js');
require('mathjax-full/js/input/tex/boldsymbol/BoldsymbolConfiguration.js');
RegisterHTMLHandler(liteAdaptor());
const tex = new TeX({packages: ['base', 'ams', 'boldsymbol'],
  formatError(_jax, error) { throw error; }});
const document = mathjax.document('', {InputJax: tex, OutputJax: new SVG()});
const visitor = new SerializedMmlVisitor();
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
function mml(source, display) {
  tex.reset();
  const result = visitor.visitTree(document.convert(source, {display, end: STATE.COMPILED}));
  if (result.includes('<merror')) throw Error('MathML error');
  return result;
}
const apiDir = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
if (fs.existsSync(output)) throw Error('Use a fresh output file');
const baseline = JSON.parse(fs.readFileSync(path.join(__dirname, 'baseline.json'), 'utf8'));
const api = JSON.parse(fs.readFileSync(path.join(apiDir, 'receipt.json'), 'utf8'));
const source = JSON.parse(execFileSync('python3', ['scripts/check_book_math.py', '--json'], {encoding: 'utf8'}));
const names = Object.keys(baseline.files);
const actual = new Map();
const decodedHtml = new Map();
for (const request of api.requests) {
  const bytes = fs.readFileSync(path.join(apiDir, request.response_file));
  if (sha(bytes) !== request.response_sha256) throw Error('Response identity mismatch');
  const dom = new JSDOM(bytes.toString());
  let current;
  for (const element of dom.window.document.querySelectorAll('h1, math-renderer')) {
    if (element.tagName === 'H1' && element.textContent.startsWith('DONGXI-MATH-AUDIT-FILE-')) {
      current = Number(element.textContent.slice('DONGXI-MATH-AUDIT-FILE-'.length));
      actual.set(current, []);
    } else if (element.tagName === 'MATH-RENDERER') {
      const temporary = dom.window.document.implementation.createHTMLDocument('');
      temporary.open();
      temporary.write(`<html><body>${element.textContent}</body></html>`);
      temporary.close();
      const decoded = temporary.body.textContent.trim().replace(/^\${1,2}|\${1,2}$/g, '');
      actual.get(current).push({tex: decoded, display: element.classList.contains('js-display-math')});
    }
  }
  dom.window.close();
}
const rows = [];
let sourcePass = 0, apiPass = 0;
for (let fileIndex = 0; fileIndex < names.length; fileIndex++) {
  const name = names[fileIndex];
  const old = baseline.files[name].expressions;
  const now = source.find(file => file.path === name).expressions;
  const rendered = actual.get(fileIndex);
  if (old.length !== now.length || old.length !== rendered.length) throw Error(`Count mismatch: ${name}`);
  old.forEach((expression, index) => {
    const row = {path: name, index, baseline_tex_sha256: sha(expression.tex),
      source_tex_sha256: sha(now[index].tex), client_tex_sha256: sha(rendered[index].tex)};
    const expected = mml(expression.tex, expression.display);
    row.baseline_mathml_sha256 = sha(expected);
    row.source_mathml_equal = now[index].display === expression.display && mml(now[index].tex, now[index].display) === expected;
    if (row.source_mathml_equal) sourcePass++;
    try {
      const clientMath = mml(rendered[index].tex, rendered[index].display);
      row.client_mathml_sha256 = sha(clientMath);
      row.client_mathml_equal = rendered[index].display === expression.display && clientMath === expected;
    } catch (error) {
      row.client_mathml_equal = false;
      row.client_error = error.message || String(error);
    }
    if (row.client_mathml_equal) apiPass++;
    else row.client_tex = rendered[index].tex;
    rows.push(row);
  });
}
const result = {utc: new Date().toISOString(),
  scope: 'Original versus repaired MathML and actual GitHub API HTML through the observed client document.write decode, emulated by jsdom27; pinned local MathJax3.2.2. Not live browser layout or GitHub deployed MathJax.',
  script_sha256: sha(fs.readFileSync(__filename)),
  api_receipt_sha256: sha(fs.readFileSync(path.join(apiDir, 'receipt.json'))),
  client_source_reference: 'client-serialization-review.json',
  total_expressions: rows.length, exact_source_mathml_matches: sourcePass,
  exact_client_mathml_matches: apiPass, rows};
fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({expressions: rows.length, sourceMathML: sourcePass, clientMathML: apiPass}));
process.exitCode = sourcePass === rows.length && apiPass === rows.length ? 0 : 1;

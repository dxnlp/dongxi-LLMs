// Unit-test the local controls with DOM stubs, not a browser rendering claim.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const code = fs.readFileSync(path.join(__dirname, 'review-player.js'), 'utf8');
for (const reduced of [false, true]) {
  const media = { matches: reduced, addEventListener: (_, fn) => media.change = fn };
  const figures = Array.from({length: 7}, (_, i) => {
    const image = {dataset: {gif: `${i}.gif`, still: `${i}.png`}, src: `${i}.png`};
    const attrs = {};
    const button = {hidden: true, setAttribute: (k,v) => attrs[k] = v,
      getAttribute: k => attrs[k], addEventListener: (_, fn) => button.click = fn};
    return {image, button, querySelector: tag => tag === 'img' ? image : button};
  });
  vm.runInNewContext(code, {
    window: {matchMedia: query => {
      assert.equal(query, '(prefers-reduced-motion: reduce)');
      return media;
    }}, document: {querySelectorAll: () => figures}
  });
  for (const [i, figure] of figures.entries()) {
    assert.equal(figure.image.src, `${i}.${reduced ? 'png' : 'gif'}`);
    assert.equal(figure.button.hidden, false);
    figure.button.click();
    assert.equal(figure.image.src, `${i}.${reduced ? 'gif' : 'png'}`);
    figure.button.click();
    assert.equal(figure.image.src, `${i}.${reduced ? 'png' : 'gif'}`);
  }
  media.matches = true; media.change();
  assert.ok(figures.every(f => f.image.src.endsWith('.png')));
  media.matches = false; media.change();
  assert.ok(figures.every(f => f.image.src.endsWith('.gif')));
}
console.log('PASS: 7 controls, toggles, reduced-motion startup and preference changes. DOM stubs only.');

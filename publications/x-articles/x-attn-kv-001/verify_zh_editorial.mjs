/** Static-only editorial checks. Does not launch or control a browser. */
import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const base = new URL('./', import.meta.url);
const read = p => readFile(new URL(p, base), 'utf8');
const source = await read('zh/x-editor-draft-body-with-image-placeholders.md');
const prior = await read('zh/revisions/2026-09-12-before-story-led.md');
const clean = await read('zh/x-editor-clean-body.html');
const review = await read('zh/review.html');
const plan = JSON.parse(await read('zh/x-editor-clean-body-image-plan.json'));
const metadata = JSON.parse(await read('metadata.json'));
assert.equal(plan.title, source.split('\n')[0].slice(2));
assert.equal(plan.image_count, 7);
assert.equal(plan.missing_image_count, 0);
assert.deepEqual(plan.images.map(i => i.path_relative.split('/').at(-1)), metadata.inline_image_order);
assert.equal(plan.images.filter(i => i.path.endsWith('.gif')).length, 5);
assert.equal((review.match(/<figure class="equation">/g) || []).length, 2);
assert.equal((review.match(/<figure class="animation">/g) || []).length, 5);
assert.equal((review.match(/<img /g) || []).length, 8);
assert(!/不是|而是|并非|契约|借口|接口/.test(source));
assert(!/https?:\/\/|\[[^\]]+\]\([^)]+\)/.test(source));
assert(!/\[IMAGE\s+\d+\]|<h1>|<img|<video|<pre|<a\b/.test(clean));
assert(!/output_|q_t|K_≤|V_≤|H_KV|\$\$|\\frac/.test(source));
assert(!/一个小实验|1\.80691|40\/12/.test(source));
assert.equal((source.match(/^\[IMAGE \d{2}\]/gm) || []).length, 7);
for (const slot of plan.images) {
  assert.equal(clean.split(slot.target_block_html).length, 2, 'Ambiguous image anchor');
  assert(slot.next_text_full && slot.target_after_text_full);
}
for (const phrase of ['项目文档', '100 万元', '50 万元', '实际 token prefix',
  '近似计算', '没有独立复现', '推理设置保持一致', '192 MiB', '384 MiB',
  '96 MiB', '**保存什么？**', '**从哪里生成？**', '**什么条件下能复用？**',
  '**减少了哪一项开销？**']) assert(source.includes(phrase), phrase);
for (const phrase of ['prefill 是处理 prompt 的计算阶段', 'KV cache 是保存下来的 K/V 状态',
  'prefix caching 是复用已有前缀状态的策略', 'vLLM 官方', '年报或软件手册',
  '不会省掉新回答的生成计算']) assert(source.includes(phrase), phrase);
assert(!/进一步理解：三个容易混淆的细节|不同 context，第一层投影|有了 cache，就不用 mask|请求结束，显存为什么/.test(source));
assert(!/<details|<video|配套动画/.test(review));
// Reused visuals must remain byte-identical to their approved render manifests.
for (const asset of metadata.files) {
  const bytes = await readFile(new URL(asset.path, base));
  assert.equal(createHash('sha256').update(bytes).digest('hex'), asset.sha256, asset.path);
}
assert.deepEqual(metadata.files.find(f => f.path === metadata.cover.path).dimensions, [2000, 800]);
assert(review.includes('../' + metadata.cover.path));
assert.equal(2 * 24 * 4096 * 8 * 64 * 2 / 1024 ** 2, 192);
assert.equal(2 * 24 * 8192 * 8 * 64 * 2 / 1024 ** 2, 384);
assert.equal(2 * 24 * 4096 * 4 * 64 * 2 / 1024 ** 2, 96);
function stats(text) {
  const body = text.replace(/^\[IMAGE.*$/gm, '');
  const paragraphs = body.split(/\n\s*\n/).filter(b => b && !/^(#|>|-|\d+\.)/.test(b));
  const lengths = paragraphs.map(p => p.length).sort((a,b) => a-b);
  return {cjkCharacters: (body.match(/[\u3400-\u9fff]/g) || []).length,
    proseParagraphs: paragraphs.length, medianParagraphCharacters: lengths[Math.floor(lengths.length/2)],
    longestParagraphCharacters: lengths.at(-1)};
}
const before = stats(prior), after = stats(source);
assert(after.medianParagraphCharacters < before.medianParagraphCharacters);
assert(after.longestParagraphCharacters <= 150);
const report = {status: 'passed', scope: 'Chinese editorial revision; static checks only',
  title: plan.title, textBlocks: plan.total_blocks, inlineMedia: 7, animations: 5,
  equations: 2, assetHashesUnchanged: true, coverRatio: '5:2',
  bodyLinks: 0, transferPlaceholders: 0, unrenderedMath: 0,
  before, after, browserVisualQA: 'Not performed: local URL blocked by browser security policy',
  cloudDraftUpdated: false, published: false};
await writeFile(new URL('zh/editorial-qa.json', base), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));

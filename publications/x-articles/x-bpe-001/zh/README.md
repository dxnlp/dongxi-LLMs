# X-BPE-001 中文版 — 本地 X Article 包

状态：中文版已经生成，等待编辑审阅。内容尚未传入 X，也没有发布。

拟定标题：**Token 到底是什么？从 Unicode 字节到 BPE 与 Token ID**

## 文件说明

- `x-editor-draft-body-with-image-placeholders.md`：中文版审阅母稿。
- `x-editor-draft-body-with-image-placeholders.html`：包含图片占位符的富文本来源。
- `x-editor-clean-body.md` 与 `.html`：去除标题和图片占位符后的 X 传输正文。
- `review.html`：包含 5:2 封面、五张行内图、图注、正文和来源链接的浏览器审阅版。
- `image-upload-order.md`：行内图片顺序。
- `x-editor-clean-body-image-plan.md` 与 `.json`：生成的倒序插图计划。
- `source-map.md`：中文版主张与本地证据的对应关系。
- `../terminology.md`：中英文版本共用的规范 NLP 术语表。
- `build_chinese_visuals.py`：中文版静态图片的可编辑源码。
- `assets/cover.png`：2000×800（5:2）中文版封面。
- `assets/01`–`05`：五张中文版行内图。
- `assets/metadata.json`：图片尺寸、哈希、字体、配色与边界说明。

## 重建

在仓库根目录运行：

```bash
python3 publications/x-articles/x-bpe-001/zh/build_chinese_visuals.py

python3 publications/x-articles/x-bpe-001/zh/build_html.py

python3 ~/.codex/skills/x-article-drafter/scripts/build_x_article_body_plan.py \
  publications/x-articles/x-bpe-001/zh
```

中文版正文与图片使用宋体；单独的英文字符串使用 Arial。封面遵循仓库统一的
5:2 规则，行内图片保留适合机制展示的 16:9 比例。

## 审阅边界

审阅应重点检查术语、中文节奏、证据边界、封面与行内图。最终标题、封面、
X 草稿传输和发布仍由用户决定。

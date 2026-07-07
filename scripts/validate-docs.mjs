#!/usr/bin/env node
// 文档命名与结构校验脚本。
// 校验规则（对应 docs/00-元规范/002-目录与命名规范.md）：
//   1. docs/ 下的分类目录必须命名为「两位数序号-分类名」，如 03-深度学习基础
//   2. 分类目录内的 .md 文件必须以三位序号开头，如 001-xxx.md
//   3. 每个分类目录必须包含一篇 000- 开头的总览/知识图谱文档
//   4. 000 总览应引用该分类下全部正文文档（001 及以后），避免新增章节后遗漏
//
// 用法：node scripts/validate-docs.mjs   （或 npm run docs:validate）
// 退出码非 0 表示校验失败，可被 Git pre-commit 钩子拦截提交。

import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const DOCS = path.join(ROOT, 'docs')

// 非分类目录/文件，跳过校验
const IGNORE_DIRS = new Set(['.vitepress', 'public', 'node_modules'])

const CATEGORY_RE = /^\d{2}-.+/ // 两位数序号-分类名
const DOC_RE = /^\d{3}-.+\.md$/i // 三位数序号-知识点.md

const errors = []
const warnings = []

if (!fs.existsSync(DOCS)) {
  console.error(`✖ 未找到 docs 目录：${DOCS}`)
  process.exit(1)
}

const entries = fs.readdirSync(DOCS, { withFileTypes: true })
const categoryDirs = entries.filter(
  (e) => e.isDirectory() && !IGNORE_DIRS.has(e.name)
)

if (categoryDirs.length === 0) {
  warnings.push('docs/ 下暂无任何分类目录（尚未开始整理文档）。')
}

for (const dir of categoryDirs) {
  // 规则 1：分类目录命名
  if (!CATEGORY_RE.test(dir.name)) {
    errors.push(
      `分类目录命名不规范："${dir.name}"，应为「两位数序号-分类名」，如 03-深度学习基础。`
    )
    continue
  }

  const catPath = path.join(DOCS, dir.name)
  const files = fs
    .readdirSync(catPath)
    .filter((f) => f.toLowerCase().endsWith('.md'))

  // 规则 3：必须包含 000- 总览文档
  const summaryFile = files.find((f) => /^000-.+\.md$/i.test(f))
  if (!summaryFile) {
    errors.push(
      `分类「${dir.name}」缺少总览/知识图谱文档（需要一篇 000- 开头的 .md）。`
    )
  }

  // 规则 2：文档文件命名
  for (const f of files) {
    if (!DOC_RE.test(f)) {
      errors.push(
        `文档命名不规范："${dir.name}/${f}"，应以三位序号开头，如 001-神经网络结构.md。`
      )
    }
  }

  // 规则 4：000 总览应引用该分类全部正文（001+）
  if (summaryFile) {
    const summaryPath = path.join(catPath, summaryFile)
    const summaryContent = fs.readFileSync(summaryPath, 'utf8')
    const articles = files.filter(
      (f) => /^(\d{3})-.+\.md$/i.test(f) && !/^000-/i.test(f)
    )

    for (const article of articles) {
      const num = article.slice(0, 3)
      // 总览中应出现「NNN-」形式的链接或文件名片段（如 ./001-xxx.md）
      if (!summaryContent.includes(`${num}-`)) {
        errors.push(
          `分类「${dir.name}」的 000 总览未引用正文「${article}」（请在 000 文档中加入链接或更新知识图谱）。`
        )
      }
    }
  }
}

// 输出结果
for (const w of warnings) console.warn(`⚠ ${w}`)

if (errors.length > 0) {
  console.error('\n✖ 文档规范校验未通过：')
  for (const e of errors) console.error(`  - ${e}`)
  console.error(`\n共 ${errors.length} 个问题，请修复后重试。`)
  process.exit(1)
}

console.log('✔ 文档规范校验通过。')
process.exit(0)

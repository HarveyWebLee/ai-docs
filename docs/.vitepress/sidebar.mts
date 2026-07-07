import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

// docs 根目录（.vitepress 的上一级）
const DOCS_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')

// 需要忽略的目录/文件
const IGNORE = new Set(['.vitepress', 'public', 'index.md', 'node_modules'])

/**
 * 从文件/目录名中提取用于展示的标题：
 * - 去掉排序用的数字前缀（如 "001-" / "01-"）
 * - 去掉 .md 扩展名
 * 例如：001-神经网络结构.md -> 神经网络结构
 */
function toTitle(name: string): string {
  return name.replace(/\.md$/i, '').replace(/^\d+[-_.]/, '')
}

/** 读取某个分类目录下的文档，按文件名（含数字前缀）自然排序 */
function readCategory(dirAbs: string, dirRel: string) {
  const items = fs
    .readdirSync(dirAbs)
    .filter((f) => f.toLowerCase().endsWith('.md') && f.toLowerCase() !== 'index.md')
    .sort((a, b) => a.localeCompare(b, 'zh-Hans-CN', { numeric: true }))

  return items.map((file) => ({
    text: toTitle(file),
    // VitePress 链接不含扩展名
    link: `/${dirRel}/${file.replace(/\.md$/i, '')}`
  }))
}

/**
 * 生成 VitePress 侧边栏配置。
 * 约定：docs 下每个「NN-分类名」目录即一个系统性分类，
 * 目录内以数字前缀命名的 .md 文件按序展示。
 */
export function generateSidebar() {
  const categories = fs
    .readdirSync(DOCS_ROOT, { withFileTypes: true })
    .filter((d) => d.isDirectory() && !IGNORE.has(d.name))
    .sort((a, b) => a.name.localeCompare(b.name, 'zh-Hans-CN', { numeric: true }))

  return categories.map((cat) => ({
    text: toTitle(cat.name),
    collapsed: false,
    items: readCategory(path.join(DOCS_ROOT, cat.name), cat.name)
  }))
}

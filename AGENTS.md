# AGENTS.md

本文件面向在本仓库中工作的 AI Agent 与贡献者，定义仓库定位、文档规范与开发/运行方式。**所有文档整理必须严格遵循下述规范。**

## 仓库定位

`ai-docs` 是一个用于**系统性学习 AI 开发**的知识库，以文档为主，配套少量用于学习/测试的代码。文档站点基于 VitePress 构建，支持中文本地化、本地搜索、LaTeX 公式与 Mermaid 知识图谱。**目标读者是 AI 开发零基础的初学者，"小白能否读懂"是最高优先级。**

## 文档核心规范（硬性要求）

编写或修改任何文档时，必须同时满足以下六条（权威版本见 `docs/00-元规范/`）：

1. **系统性分类**：文档必须归入 `docs/` 下某个「两位数序号-分类名」目录（如 `03-深度学习基础`），不允许游离文档。
2. **章节序号命名**：分类内文档以三位序号开头，如 `001-神经网络结构.md`。
3. **全面且正确**：知识点需覆盖核心要素，公式/结论/数据必须**可追溯、可验证、正确无误**。
4. **教授级讲解 + 案例**：以大学教授级严谨度讲解，并配可复现的真实案例。
5. **通俗优先、专业对齐（头等要求）**：面向零基础读者，**每个专业概念/专有名词都必须先用大白话或类比讲清"是什么、为什么、解决什么问题"，再给严谨定义，最后把专业表述与前面的通俗讲法显式对齐**（"这个公式说的就是前面那句大白话"）。严禁一上来就甩定义、公式、术语。术语要"先解释再使用"，多用比喻/例子/对比表格。
6. **分类总结与知识图谱**：每个分类必须含一篇 `000-` 开头的总览文档，串联知识点并用 Mermaid 绘制知识图谱。

单篇文档推荐结构（**顺序原则：先通俗 → 后专业 → 再对齐**）：`一句话先说清 → 打个比方 → 它到底解决什么问题（分问题+对比表格）→ 专业视角（与大白话对齐）→ 案例解析 → 常见误区 → 一句话总结`。详见 `docs/00-元规范/001-文档编写规范.md`；**通俗写作黄金样板见 `docs/03-深度学习基础/001-神经网络结构.md` 的"为什么必须有非线性激活函数"一节。**

## 目录约定

- 文档：`docs/NN-分类名/NNN-知识点.md`
- 配套代码：`code/NN-分类名/NNN-知识点/`（镜像文档目录）
- 校验脚本：`scripts/validate-docs.mjs`
- 完整说明见 `docs/00-元规范/002-目录与命名规范.md`。

## 开发与运行（标准命令见 package.json）

- 安装依赖：`npm install`
- 本地预览（开发服务器）：`npm run dev`（默认 http://localhost:5173）
- 生产构建：`npm run build`；本地预览构建产物：`npm run preview`
- 文档规范校验（即本仓库的 lint）：`npm run docs:validate`（等价于 `npm run lint`）

提交前请运行 `npm run docs:validate` 并本地 `npm run dev` 预览，确认公式、知识图谱与内部链接均正常。Git `pre-commit` 钩子会自动运行校验。Push/PR 时 GitHub Actions（`.github/workflows/ci.yml`）会执行 `docs:validate`、`build` 及 `code/` 下全部 Python 示例。

## MCP 服务

`.cursor/mcp.json` 预置了两个**免费公共** MCP 服务，用于查资料、保证正确性：
- `deepwiki`：查询 GitHub 开源项目文档（免鉴权）。
- `context7`：查询各类库/框架的最新官方文档（免费额度内免鉴权）。

## Cursor Cloud specific instructions

- 站点为 VitePress。开发服务器用 `npm run dev`，默认监听 `http://localhost:5173`；如需对外访问加 `-- --host`。
- LaTeX 公式依赖 `markdown-it-mathjax3` 且需在 `docs/.vitepress/config.mts` 中开启 `markdown.math: true`；知识图谱依赖 `vitepress-plugin-mermaid`（配置用 `withMermaid` 包装）。二者已配置，改动配置后需重启 dev server 才生效。
- 侧边栏由 `docs/.vitepress/sidebar.mts` 依据目录命名**自动生成**，因此新增文档必须遵守「NN-分类名 / NNN-知识点.md」命名，否则不会出现在侧边栏或排序错乱。
- `ignoreDeadLinks: false`：文档内部链接失效会导致 `npm run build` 失败，新增/移动文档时需同步更新引用。
- 配套 Python 示例仅依赖标准库，可直接 `python3 code/.../xor.py` 运行，无需虚拟环境。

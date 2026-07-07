# ai-docs

用于**系统性学习 AI 开发**的知识库：以文档为主，配套少量学习/测试代码。文档站点基于 [VitePress](https://vitepress.dev/) 构建，支持中文本地化、本地搜索、LaTeX 公式与 Mermaid 知识图谱。

## 快速开始

```bash
npm install          # 安装依赖
npm run dev          # 启动开发服务器（默认 http://localhost:5173）
npm run build        # 生产构建
npm run preview      # 预览构建产物
npm run docs:validate # 校验文档命名与结构（lint）
```

## 目录结构

```text
ai-docs/
├── docs/                 # 文档站点根目录
│   ├── .vitepress/       # 站点配置与侧边栏生成器
│   ├── 00-元规范/        # 文档规范（分类、命名、写作标准）
│   └── NN-分类名/        # 各系统性分类，内含 000 总览 + NNN 知识点
├── code/                 # 配套学习/测试代码（镜像文档分类）
├── scripts/              # 校验脚本
├── .cursor/              # Cursor 规则、skills、MCP 配置
└── AGENTS.md             # 面向 AI Agent 的仓库约定
```

## 文档规范

所有文档遵循统一规范（系统性分类、章节序号命名、教授级讲解 + 通俗解读 + 案例、分类知识图谱）。详见 `AGENTS.md` 与 `docs/00-元规范/`。

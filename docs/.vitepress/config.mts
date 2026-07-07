import { withMermaid } from 'vitepress-plugin-mermaid'
import { generateSidebar } from './sidebar.mts'

// 使用 withMermaid 包装配置，使 ```mermaid 代码块可直接渲染为知识图谱
export default withMermaid({
  // 站点语言：默认中文
  lang: 'zh-CN',
  title: 'AI 开发学习知识库',
  description: '系统化分类的 AI 开发学习文档，配套学习/测试代码与知识图谱',

  // 关闭死链检查会掩盖问题；开启后构建时校验文档内部链接是否有效
  ignoreDeadLinks: false,

  // 兼容中文标题生成的锚点
  markdown: {
    lineNumbers: true,
    // 启用 LaTeX 数学公式渲染（依赖 markdown-it-mathjax3）
    math: true
  },

  themeConfig: {
    // 顶部导航
    nav: [
      { text: '首页', link: '/' },
      { text: '文档规范', link: '/00-元规范/000-分类总览与知识图谱' }
    ],

    // 侧边栏：由 docs 目录结构自动生成（严格依赖编号命名规范排序）
    sidebar: generateSidebar(),

    // 中文界面文案
    outline: { label: '本页目录', level: [2, 3] },
    docFooter: { prev: '上一篇', next: '下一篇' },
    darkModeSwitchLabel: '外观',
    returnToTopLabel: '返回顶部',
    sidebarMenuLabel: '目录',

    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档', buttonAriaLabel: '搜索文档' },
          modal: {
            noResultsText: '无法找到相关结果',
            resetButtonTitle: '清除查询条件',
            footer: {
              selectText: '选择',
              navigateText: '切换',
              closeText: '关闭'
            }
          }
        }
      }
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/HarveyWebLee/ai-docs' }
    ]
  }
})

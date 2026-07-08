import DefaultTheme from 'vitepress/theme'
import { setupMermaidZoom } from './mermaid-zoom'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ router }) {
    // 仅客户端：为 Mermaid 图谱绑定点击放大预览
    if (typeof window === 'undefined') return

    setupMermaidZoom((cb) => {
      const previous = router.onAfterRouteChanged
      router.onAfterRouteChanged = () => {
        previous?.()
        cb()
      }
    })
  }
}

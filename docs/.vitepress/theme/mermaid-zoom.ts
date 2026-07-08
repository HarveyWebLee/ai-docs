/**
 * 为 VitePress 页面中的 Mermaid 图谱提供「点击放大 + 滚轮缩放 + 拖拽平移」。
 * 缩放/平移通过 SVG viewBox 实现（矢量重绘），避免 CSS transform 放大导致发糊。
 */

const ENHANCED_ATTR = 'data-mermaid-zoom'
const BASE_VB_KEY = 'data-base-view-box'

/** viewBox：x/y 为左上角，w/h 为可视区域宽高（越小越「放大」） */
interface ViewBox {
  x: number
  y: number
  w: number
  h: number
}

const MIN_ZOOM = 0.25
const MAX_ZOOM = 8
const ZOOM_FACTOR = 1.12

let overlay: HTMLElement | null = null
let inner: HTMLElement | null = null
let stage: HTMLElement | null = null
let activeSvg: SVGSVGElement | null = null
let dragging = false
let dragStart = { x: 0, y: 0, vb: null as ViewBox | null }

function parseViewBox(svg: SVGSVGElement): ViewBox {
  const vb = svg.viewBox?.baseVal
  if (vb && vb.width > 0 && vb.height > 0) {
    return { x: vb.x, y: vb.y, w: vb.width, h: vb.height }
  }
  const bbox = svg.getBBox()
  return { x: bbox.x, y: bbox.y, w: bbox.width || 1, h: bbox.height || 1 }
}

function readBaseViewBox(svg: SVGSVGElement): ViewBox {
  const raw = svg.getAttribute(BASE_VB_KEY)
  if (raw) return JSON.parse(raw) as ViewBox
  return parseViewBox(svg)
}

function applyViewBox(svg: SVGSVGElement, vb: ViewBox) {
  svg.setAttribute('viewBox', `${vb.x} ${vb.y} ${vb.w} ${vb.h}`)
}

function currentZoom(svg: SVGSVGElement): number {
  const base = readBaseViewBox(svg)
  const cur = parseViewBox(svg)
  return base.w / cur.w
}

function prepareSvg(svg: SVGSVGElement) {
  const base = parseViewBox(svg)
  svg.setAttribute(BASE_VB_KEY, JSON.stringify(base))
  svg.setAttribute('preserveAspectRatio', 'xMidYMid meet')
  svg.style.maxWidth = 'none'
  svg.style.width = '100%'
  svg.style.height = '100%'
  svg.style.display = 'block'
  applyViewBox(svg, base)
}

/** 屏幕坐标 → SVG 用户坐标（用于以鼠标为中心缩放） */
function clientToSvg(svg: SVGSVGElement, clientX: number, clientY: number) {
  const pt = svg.createSVGPoint()
  pt.x = clientX
  pt.y = clientY
  const ctm = svg.getScreenCTM()
  if (!ctm) return { x: 0, y: 0 }
  const p = pt.matrixTransform(ctm.inverse())
  return { x: p.x, y: p.y }
}

function zoomBy(factor: number, clientX?: number, clientY?: number) {
  if (!activeSvg || !stage || factor <= 0) return

  const cur = parseViewBox(activeSvg)
  const base = readBaseViewBox(activeSvg)
  const nextW = cur.w / factor
  const nextH = cur.h / factor
  const nextZoom = base.w / nextW

  if (nextZoom < MIN_ZOOM || nextZoom > MAX_ZOOM) return

  let fx: number
  let fy: number
  if (clientX != null && clientY != null) {
    const p = clientToSvg(activeSvg, clientX, clientY)
    fx = p.x
    fy = p.y
  } else {
    fx = cur.x + cur.w / 2
    fy = cur.y + cur.h / 2
  }

  applyViewBox(activeSvg, {
    x: fx - (fx - cur.x) / factor,
    y: fy - (fy - cur.y) / factor,
    w: nextW,
    h: nextH
  })
}

function panByPixels(dx: number, dy: number) {
  if (!activeSvg || !stage) return
  const cur = parseViewBox(activeSvg)
  const rect = stage.getBoundingClientRect()
  if (rect.width === 0 || rect.height === 0) return

  applyViewBox(activeSvg, {
    x: cur.x - (dx * cur.w) / rect.width,
    y: cur.y - (dy * cur.h) / rect.height,
    w: cur.w,
    h: cur.h
  })
}

function resetView() {
  if (!activeSvg) return
  applyViewBox(activeSvg, readBaseViewBox(activeSvg))
}

function openOverlay(source: HTMLElement) {
  if (!overlay || !inner || !stage) return

  const svg = source.querySelector('svg')
  if (!svg) return

  inner.replaceChildren(svg.cloneNode(true) as Node)
  activeSvg = inner.querySelector('svg')
  if (!activeSvg) return

  prepareSvg(activeSvg)
  overlay.classList.remove('hidden')
  document.body.classList.add('mermaid-zoom-open')
}

function closeOverlay() {
  if (!overlay) return
  overlay.classList.add('hidden')
  document.body.classList.remove('mermaid-zoom-open')
  if (inner) inner.replaceChildren()
  activeSvg = null
  dragging = false
}

function ensureOverlay() {
  if (overlay) return

  overlay = document.createElement('div')
  overlay.id = 'mermaid-zoom-overlay'
  overlay.className = 'mermaid-zoom-overlay hidden'
  overlay.setAttribute('role', 'dialog')
  overlay.setAttribute('aria-modal', 'true')
  overlay.setAttribute('aria-label', 'Mermaid 图谱放大预览')

  overlay.innerHTML = `
    <div class="mermaid-zoom-backdrop" data-action="close"></div>
    <div class="mermaid-zoom-panel">
      <div class="mermaid-zoom-toolbar">
        <span class="mermaid-zoom-title">图谱预览</span>
        <div class="mermaid-zoom-actions">
          <button type="button" class="mermaid-zoom-btn" data-action="zoom-out" title="缩小">−</button>
          <button type="button" class="mermaid-zoom-btn" data-action="zoom-in" title="放大">+</button>
          <button type="button" class="mermaid-zoom-btn" data-action="reset" title="重置视图">重置</button>
          <button type="button" class="mermaid-zoom-btn mermaid-zoom-btn-close" data-action="close" title="关闭 (Esc)">关闭</button>
        </div>
      </div>
      <div class="mermaid-zoom-stage">
        <div class="mermaid-zoom-inner"></div>
      </div>
      <p class="mermaid-zoom-hint">滚轮缩放 · 拖拽平移 · Esc 关闭（矢量清晰缩放）</p>
    </div>
  `

  document.body.appendChild(overlay)
  inner = overlay.querySelector('.mermaid-zoom-inner')
  stage = overlay.querySelector('.mermaid-zoom-stage')

  overlay.addEventListener('click', (e) => {
    const target = e.target as HTMLElement
    const action = target.closest('[data-action]')?.getAttribute('data-action')
    if (action === 'close') closeOverlay()
    if (action === 'zoom-in') zoomBy(ZOOM_FACTOR)
    if (action === 'zoom-out') zoomBy(1 / ZOOM_FACTOR)
    if (action === 'reset') resetView()
  })

  stage?.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault()
      const factor = e.deltaY < 0 ? ZOOM_FACTOR : 1 / ZOOM_FACTOR
      zoomBy(factor, e.clientX, e.clientY)
    },
    { passive: false }
  )

  stage?.addEventListener('mousedown', (e) => {
    if (e.button !== 0 || !activeSvg) return
    dragging = true
    stage!.classList.add('is-dragging')
    dragStart = { x: e.clientX, y: e.clientY, vb: parseViewBox(activeSvg) }
    e.preventDefault()
  })

  window.addEventListener('mousemove', (e) => {
    if (!dragging || !activeSvg || !dragStart.vb || !stage) return
    const dx = e.clientX - dragStart.x
    const dy = e.clientY - dragStart.y
    const rect = stage.getBoundingClientRect()
    const vb = dragStart.vb
    applyViewBox(activeSvg, {
      x: vb.x - (dx * vb.w) / rect.width,
      y: vb.y - (dy * vb.h) / rect.height,
      w: vb.w,
      h: vb.h
    })
  })

  window.addEventListener('mouseup', () => {
    if (!dragging) return
    dragging = false
    stage?.classList.remove('is-dragging')
    dragStart.vb = null
  })

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && overlay && !overlay.classList.contains('hidden')) {
      closeOverlay()
    }
  })
}

function enhanceMermaid(el: HTMLElement) {
  if (el.hasAttribute(ENHANCED_ATTR)) return
  if (!el.querySelector('svg')) return

  el.setAttribute(ENHANCED_ATTR, 'true')
  el.classList.add('mermaid-zoomable')
  el.setAttribute('role', 'button')
  el.setAttribute('tabindex', '0')
  el.setAttribute('title', '点击放大查看（滚轮缩放、拖拽平移）')

  const open = () => openOverlay(el)
  el.addEventListener('click', open)
  el.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      open()
    }
  })
}

function enhanceAll() {
  document.querySelectorAll<HTMLElement>('div.mermaid').forEach(enhanceMermaid)
}

/** 在客户端初始化；由 theme/index.ts 调用 */
export function setupMermaidZoom(onRouteChange: (fn: () => void) => void) {
  if (typeof window === 'undefined') return

  ensureOverlay()
  enhanceAll()

  onRouteChange(() => {
    enhanceAll()
    window.setTimeout(enhanceAll, 150)
    window.setTimeout(enhanceAll, 600)
  })

  const root = document.querySelector('.vp-doc') ?? document.body
  const observer = new MutationObserver(() => enhanceAll())
  observer.observe(root, { childList: true, subtree: true })
}

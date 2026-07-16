# FGT Server Dashboard - Frontend Implementation Plan

## Overview

The FGT Server Dashboard is a single-page Neumorphic (Soft UI) web interface served as static files by the FastAPI server. It provides real-time monitoring and control of the federated learning session. Built with **vanilla HTML + CSS + JS** (no frameworks), using **Chart.js** for data visualization and native **WebSocket API** for live updates.

**Files produced:**
```
server/dashboard/
    index.html        # Structure + semantic markup
    styles.css        # Complete Neumorphic design system
    dashboard.js      # WebSocket client + Chart.js + DOM management
```

---

## 1. Design System

### 1.1 Color Palette

| Token | Value | Usage |
|:---|:---|:---|
| `--bg-base` | `#F0E4D7` | Page background, panel backgrounds |
| `--bg-surface` | `#EAD9C8` | Slightly darker surface for depth layering |
| `--bg-sidebar` | `#E5D3C1` | Sidebar background |
| `--shadow-light` | `#FDFAF6` | Light shadow (top-left, simulates light source) |
| `--shadow-dark` | `#C9B9A5` | Dark shadow (bottom-right, simulates depth) |
| `--shadow-light-strong` | `#FFFFFF` | Stronger light shadow for elevated elements |
| `--shadow-dark-strong` | `#B5A48E` | Stronger dark shadow for elevated elements |
| `--accent-primary` | `#D4845A` | Terracotta — primary actions, active states |
| `--accent-gold` | `#E8B87A` | Warm gold — highlights, charts line 1 |
| `--accent-deep` | `#8B5E3C` | Deep warm brown — secondary actions |
| `--accent-rose` | `#C97B7B` | Muted rose — charts line 2, warnings |
| `--accent-sage` | `#7A9E7E` | Muted sage green — success states, charts line 3 |
| `--text-primary` | `#3D2B1F` | Headings, primary text |
| `--text-secondary` | `#7A6355` | Labels, secondary text |
| `--text-muted` | `#A89485` | Placeholders, disabled text |
| `--success` | `#6B8F5E` | Online indicators, positive deltas |
| `--warning` | `#C4954A` | Caution states |
| `--error` | `#B85C4A` | Offline indicators, negative deltas |

### 1.2 Shadow System

All neumorphic effects derive from these shadow primitives:

```css
/* Raised (default resting state) */
--nm-raised: 6px 6px 12px var(--shadow-dark),
            -6px -6px 12px var(--shadow-light);

/* Raised Strong (elevated cards, modals) */
--nm-raised-strong: 8px 8px 16px var(--shadow-dark-strong),
                   -8px -8px 16px var(--shadow-light-strong);

/* Pressed (active/clicked state) */
--nm-pressed: inset 4px 4px 8px var(--shadow-dark),
              inset -4px -4px 8px var(--shadow-light);

/* Concave (input fields, inset containers) */
--nm-concave: inset 3px 3px 6px var(--shadow-dark),
              inset -3px -3px 6px var(--shadow-light);

/* Flat (hover transition state — no shadow) */
--nm-flat: none;
```

**Rule:** Element `background-color` must always match its parent container's `background-color`. Breaking this rule destroys the neumorphic illusion.

### 1.3 Typography

| Role | Font | Weight | Size | Line Height |
|:---|:---|:---|:---|:---|
| H1 (Page title) | Inter | 700 | 28px | 1.3 |
| H2 (Panel title) | Inter | 600 | 20px | 1.3 |
| H3 (Section label) | Inter | 600 | 16px | 1.4 |
| Body | Outfit | 400 | 14px | 1.5 |
| Body Small | Outfit | 400 | 12px | 1.5 |
| Label | Outfit | 500 | 12px | 1.4 |
| Mono (metrics) | JetBrains Mono | 400 | 14px | 1.4 |
| Mono Small | JetBrains Mono | 400 | 12px | 1.4 |

**Google Fonts import:** `Inter:wght@400;500;600;700`, `Outfit:wght@400;500`, `JetBrains+Mono:wght@400`

### 1.4 Spacing Scale

Base unit: `4px`. Scale: `4, 8, 12, 16, 20, 24, 32, 40, 48, 64`

| Token | Value | Usage |
|:---|:---|:---|
| `--space-xs` | `4px` | Inline spacing, icon gaps |
| `--space-sm` | `8px` | Tight padding |
| `--space-md` | `16px` | Standard padding, element gaps |
| `--space-lg` | `24px` | Panel padding |
| `--space-xl` | `32px` | Section gaps |
| `--space-2xl` | `48px` | Major section separation |

### 1.5 Border Radius

| Token | Value | Usage |
|:---|:---|:---|
| `--radius-sm` | `8px` | Chips, small buttons |
| `--radius-md` | `12px` | Cards, inputs |
| `--radius-lg` | `16px` | Panels, large containers |
| `--radius-xl` | `24px` | Large feature cards |
| `--radius-pill` | `999px` | Pills, toggle tracks |

### 1.6 Transitions

All interactive state changes use:
```css
transition: box-shadow 200ms ease, background-color 200ms ease, transform 100ms ease;
```

---

## 2. Layout Architecture

### 2.1 CSS Grid Structure

```
+-------------------------------------------------------------+
|                        HEADER (60px)                         |
|  [Logo/Title]                    [Server IP] [Model v#] [*]  |
+----------+--------------------------------------------------+
|          |                                                    |
| SIDEBAR  |                  MAIN CONTENT                     |
| (220px)  |          (CSS Grid: auto-fill panels)             |
|          |                                                    |
| [Nav     |  +------------------+  +------------------+       |
|  Items]  |  | Server Status    |  | Connected Clients|       |
|          |  +------------------+  +------------------+       |
|          |                                                    |
|          |  +-------------------------------------------+    |
|          |  | Training Control                           |    |
|          |  +-------------------------------------------+    |
|          |                                                    |
|          |  +------------------+  +------------------+       |
|          |  | Accuracy Chart   |  | Loss Chart        |       |
|          |  +------------------+  +------------------+       |
|          |                                                    |
|          |  +------------------+  +------------------+       |
|          |  | FL vs Baseline   |  | Leaderboard       |       |
|          |  +------------------+  +------------------+       |
|          |                                                    |
|          |  +------------------+  +------------------+       |
|          |  | Privacy Panel    |  | Export Panel       |       |
|          |  +------------------+  +------------------+       |
+----------+--------------------------------------------------+
```

```css
.dashboard {
  display: grid;
  height: 100vh;
  grid-template-areas:
    "sidebar header"
    "sidebar main";
  grid-template-columns: 220px 1fr;
  grid-template-rows: 60px 1fr;
  background: var(--bg-base);
  overflow: hidden;
}

.main-content {
  grid-area: main;
  padding: var(--space-lg);
  overflow-y: auto;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(380px, 1fr));
  gap: var(--space-lg);
  align-content: start;
}
```

### 2.2 Responsive Behavior

| Viewport | Behavior |
|:---|:---|
| >= 1200px | 2-column panel grid, sidebar visible |
| 900px - 1199px | 2-column panel grid, sidebar collapsed to icons (56px) |
| < 900px | 1-column panel grid, sidebar hidden (hamburger toggle) |

---

## 3. Component Library

### 3.1 NM-Card (Panel Container)

```html
<div class="nm-card" id="panel-status">
  <div class="nm-card__header">
    <h2 class="nm-card__title">Server Status</h2>
    <span class="nm-card__badge">Live</span>
  </div>
  <div class="nm-card__body">
    <!-- Panel content -->
  </div>
</div>
```

**States:**
- Default: `--nm-raised`, `--bg-base` background
- Hover: Shadow intensifies slightly (8px blur -> 10px blur)
- Loading: Content area has subtle shimmer animation

---

### 3.2 NM-Button

```html
<button class="nm-btn nm-btn--primary" id="btn-start-training">
  Start Training
</button>
```

**Variants:**
| Class | Appearance |
|:---|:---|
| `.nm-btn--primary` | Raised, `--accent-primary` text, bold |
| `.nm-btn--secondary` | Raised, `--text-secondary` text |
| `.nm-btn--danger` | Raised, `--error` text |
| `.nm-btn--icon` | Raised, circular, icon-only (40x40) |

**States:**
- Default: `--nm-raised`
- Hover: Shadow strengthens, subtle scale `transform: scale(1.01)`
- Active/Pressed: `--nm-pressed`, `transform: scale(0.98)`
- Disabled: `--nm-flat`, `opacity: 0.5`, `cursor: not-allowed`

---

### 3.3 NM-Input

```html
<div class="nm-input-group">
  <label class="nm-label" for="input-rounds">Max Rounds</label>
  <input class="nm-input" type="number" id="input-rounds" value="10">
</div>
```

**Appearance:** `--nm-concave` (inset shadow), matches background color.
**Focus state:** Inner shadow transitions to use `--accent-primary` tinted shadow, subtle border-left accent line appears.

---

### 3.4 NM-Toggle

```html
<div class="nm-toggle" id="toggle-dp" role="switch" aria-checked="true" tabindex="0">
  <div class="nm-toggle__track">
    <div class="nm-toggle__thumb"></div>
  </div>
  <span class="nm-toggle__label">Differential Privacy</span>
</div>
```

**Track:** `--nm-concave`, pill shape, 48px wide.
**Thumb:** `--nm-raised`, circular (22px), slides with `transform: translateX()`.
**On state:** Track has faint `--accent-primary` background tint.

---

### 3.5 NM-Progress

```html
<div class="nm-progress" role="progressbar" aria-valuenow="60" aria-valuemin="0" aria-valuemax="100">
  <div class="nm-progress__track">
    <div class="nm-progress__fill" style="width: 60%"></div>
  </div>
  <span class="nm-progress__label">Round 6 / 10</span>
</div>
```

**Track:** `--nm-concave`, 8px height, pill radius.
**Fill:** `--nm-raised`, gradient from `--accent-gold` to `--accent-primary`, smooth width transition.

---

### 3.6 NM-Status-Dot

```html
<span class="nm-dot nm-dot--online" aria-label="Connected"></span>
```

**Variants:**
- `.nm-dot--online` — `--success` color, subtle pulse animation (scale 1 -> 1.3, opacity 1 -> 0.4, 2s infinite)
- `.nm-dot--offline` — `--error` color, no animation
- `.nm-dot--training` — `--accent-gold` color, faster pulse (1s)

---

### 3.7 NM-Table

```html
<table class="nm-table" id="leaderboard-table">
  <thead><tr><th>Rank</th><th>Client</th><th>Score</th></tr></thead>
  <tbody><!-- Dynamic rows --></tbody>
</table>
```

**Styling:** No borders. Alternating row backgrounds using slightly different background tints. Header row uses `--text-secondary` with `font-weight: 600`. Rows have subtle `--nm-raised` on hover.

---

### 3.8 NM-Chip (Tag Category)

```html
<span class="nm-chip" data-category="places">Places</span>
```

**Appearance:** Small pill, `--nm-raised`, category-specific left border color. `font-size: 12px`.

---

### 3.9 NM-Modal (Confirmation Dialogs)

```html
<div class="nm-modal-overlay" id="modal-export" hidden>
  <div class="nm-modal nm-card">
    <h3 class="nm-modal__title">Export Model</h3>
    <p class="nm-modal__body">Download the final trained TFLite model?</p>
    <div class="nm-modal__actions">
      <button class="nm-btn nm-btn--secondary" data-action="cancel">Cancel</button>
      <button class="nm-btn nm-btn--primary" data-action="confirm">Download</button>
    </div>
  </div>
</div>
```

**Overlay:** Semi-transparent warm tint (`rgba(61, 43, 31, 0.3)`), backdrop blur (`4px`).
**Modal:** `--nm-raised-strong`, centered, max-width 440px.

---

## 4. Dashboard Panels (Detailed Specifications)

### 4.1 Server Status Panel

**Grid span:** 1 column.
**Contents:**

| Element | Type | Data Source | Update Trigger |
|:---|:---|:---|:---|
| Server Uptime | Mono text, `HH:MM:SS` | JS timer (local) | Every 1s |
| LAN IP Address | Mono text | REST `GET /api/training/status` on load | Once |
| Model Version | Badge, `Round #N` | WebSocket `round_completed` | Per round |
| Training Phase | Text + color indicator | WebSocket `round_started`, `training_complete` | Per event |
| Total Clients (ever connected) | Number | WebSocket `client_connected` | Per event |

**Phase Indicator States:**
- `Idle` — muted text, no dot
- `Waiting for Clients` — `--warning` dot, pulsing
- `Training Round N` — `--accent-primary` dot, pulsing
- `Aggregating` — `--accent-gold` dot, spinning
- `Complete` — `--success` dot, static

---

### 4.2 Connected Clients Panel

**Grid span:** 1 column. Scrollable if > 4 clients.
**Contents:** Dynamic list of client cards.

**Per-client card:**
```
+------------------------------------------+
|  [*] ClientNickname          [Online]     |
|  Device: Pixel 7a / 8GB RAM              |
|  Images: 342    Rounds: 3/5              |
|  Last seen: 2s ago                       |
+------------------------------------------+
```

- `[*]` = NM-Status-Dot (online/offline based on heartbeat)
- Cards use `.nm-card` with smaller padding
- Cards fade in with CSS `@keyframes slideIn` on connect
- Cards transition to `opacity: 0.5` on disconnect (with red dot), removed after 30s

**Data source:** WebSocket events `client_connected`, `client_disconnected`, `update_received`

---

### 4.3 Training Control Panel

**Grid span:** Full width (spans all columns via `grid-column: 1 / -1`).
**Layout:** Horizontal flex — config inputs on left, action buttons center, round progress right.

**Config Inputs (NM-Input):**

| Input | ID | Type | Default | Range |
|:---|:---|:---|:---|:---|
| Min Clients | `input-min-clients` | number | 2 | 1-20 |
| Max Rounds | `input-max-rounds` | number | 10 | 1-50 |
| Local Epochs | `input-local-epochs` | number | 3 | 1-10 |
| Learning Rate | `input-lr` | number (step 0.001) | 0.001 | 0.0001-0.1 |

**Action Buttons:**
- `Start Training` (`.nm-btn--primary`, disabled until min_clients met)
- `Stop Training` (`.nm-btn--danger`, hidden when idle, visible when training)

**Round Progress (NM-Progress):**
- Shows `Round {current} / {max}`
- Fill animates smoothly per round completion
- Below progress bar: estimated time remaining (computed from avg round duration)

**Inputs are disabled during active training.**

---

### 4.4 Metrics Panel (Accuracy + Loss Charts)

**Grid span:** 1 column each (2 charts side by side).

#### 4.4a Accuracy Chart
- **Type:** Chart.js Line Chart
- **X-Axis:** Round number (integer)
- **Y-Axis:** Accuracy (0.0 - 1.0)
- **Datasets:**
  - `Global Accuracy` — `--accent-primary` (terracotta), solid line, `tension: 0.4`
  - `Baseline Accuracy` — `--text-muted`, dashed line, static horizontal reference

**Chart.js Config:**
```javascript
{
  type: 'line',
  options: {
    animation: { duration: 500, easing: 'easeOutQuart' },
    scales: {
      y: { min: 0, max: 1, ticks: { color: 'var(--text-secondary)' } },
      x: { ticks: { color: 'var(--text-secondary)' } }
    },
    plugins: {
      legend: { labels: { color: 'var(--text-primary)', font: { family: 'Outfit' } } }
    }
  }
}
```

**Canvas container:** `.nm-card` with `--nm-concave` inset background for the chart area (gives the chart a "recessed into surface" feel).

#### 4.4b Loss Chart
- **Type:** Chart.js Line Chart
- **Datasets:**
  - `Train Loss` — `--accent-gold`, solid
  - `Validation Loss` — `--accent-rose`, solid
- Same styling/animation approach as Accuracy Chart

**Update strategy:** On WebSocket `round_completed` event, push new data point, call `chart.update()`. Sliding window of 50 data points max.

---

### 4.5 Per-Class F1 Chart

**Grid span:** Full width.

- **Type:** Chart.js Horizontal Bar Chart
- **Y-Axis:** Tag categories (from taxonomy: People, Places, Activities, ...)
- **X-Axis:** F1 Score (0.0 - 1.0)
- **Bar color:** Gradient from `--accent-gold` (low) to `--accent-sage` (high)
- **Update:** On each round completion, bars animate to new values

---

### 4.6 Comparison Panel (FL vs Baseline)

**Grid span:** 1 column.

- **Type:** Chart.js Grouped Bar Chart
- **X-Axis:** Tag categories
- **Y-Axis:** Accuracy (0.0 - 1.0)
- **Groups:**
  - `Baseline (Pre-trained)` — `--text-muted`, striped pattern fill
  - `Federated Model` — `--accent-primary`, solid fill
- **Delta labels:** Above each FL bar, show `+X.X%` improvement over baseline in `--success` color
- **Update:** Refreshed on each round completion from `/api/metrics/comparison`

---

### 4.7 Leaderboard Panel

**Grid span:** 1 column.

**Table columns:**

| Column | Width | Alignment |
|:---|:---|:---|
| Rank | 48px | Center |
| Client | 1fr | Left |
| Images | 80px | Right |
| Rounds | 64px | Right |
| Score | 80px | Right |

- Top 3 ranks get a subtle warm glow effect on their row
- Rank 1 row has a faint `--accent-gold` left border
- Sortable by clicking column headers (JS sort, no server call)
- Updated via WebSocket `round_completed` event

---

### 4.8 Privacy Panel

**Grid span:** 1 column.

**Content:** An SVG diagram (inline, hand-crafted) showing the data flow:

```
+-------------------+                    +-------------------+
|   Android Device  |   Weights Only     |   FGT Server      |
|                   | -----------------> |                   |
|  [Images STAY]    |   (DP Noise Added) |  [Aggregates]     |
|  [Tags STAY]      |                    |  [Global Model]   |
|  [Personal Data   | <----------------- |                   |
|   STAYS]          |   Updated Model    |                   |
+-------------------+                    +-------------------+
```

- Device box: `--accent-sage` border (safe/private)
- Arrow: `--accent-gold` with animated dashed stroke (CSS `stroke-dashoffset` animation, flows left-to-right)
- Server box: `--accent-primary` border
- Labels in `--text-primary`, items that stay on device in `--success`

**Below the diagram:**
- DP Epsilon value displayed as a "Privacy Strength" meter (NM-Progress, higher epsilon = lower privacy, shown inversely)
- Gradient clipping norm value
- Text: "No raw images or personal data leave your device"

---

### 4.9 Export Panel

**Grid span:** 1 column.

**Contents:**
- **Download Model** button (`.nm-btn--primary`)
  - Fetches `GET /api/export/model` -> triggers browser download of `.tflite` file
  - Shows file size estimate
  - Disabled if no training has completed
  
- **Download Training Report** button (`.nm-btn--secondary`)
  - Fetches `GET /api/export/report` -> triggers browser download of JSON report
  - Report includes FL gains comparison data
  - Disabled if no training has completed

- **Model Info** section:
  - Architecture: MobileNetV3-Small + Custom Head
  - Parameters: trainable count
  - Last updated: timestamp
  - Final accuracy: value

---

## 5. Real-Time Architecture (dashboard.js)

### 5.1 WebSocket Connection

```javascript
class FGTDashboard {
  constructor() {
    this.ws = null;
    this.reconnectDelay = 1000;
    this.maxReconnectDelay = 30000;
    this.charts = {};
    this.state = { /* app state */ };
  }

  connect() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${protocol}//${location.host}/ws/feed`);
    
    this.ws.onopen = () => {
      this.reconnectDelay = 1000;  // Reset on successful connect
      this.updateConnectionStatus('connected');
    };
    
    this.ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      this.routeEvent(msg.type, msg.data);
    };
    
    this.ws.onclose = () => {
      this.updateConnectionStatus('disconnected');
      setTimeout(() => this.connect(), this.reconnectDelay);
      this.reconnectDelay = Math.min(this.reconnectDelay * 2, this.maxReconnectDelay);
    };
  }
  
  routeEvent(type, data) {
    const handlers = {
      'client_connected':    (d) => this.addClientCard(d),
      'client_disconnected': (d) => this.removeClientCard(d),
      'round_started':       (d) => this.onRoundStart(d),
      'update_received':     (d) => this.onUpdateReceived(d),
      'round_completed':     (d) => this.onRoundComplete(d),
      'training_complete':   (d) => this.onTrainingComplete(d),
      'metrics_update':      (d) => this.updateMetrics(d),
      'error':               (d) => this.showError(d),
    };
    handlers[type]?.(data);
  }
}
```

### 5.2 Event Flow

```
WebSocket Event          ->  Handler Method        ->  DOM Updates
─────────────────────────────────────────────────────────────────
client_connected         ->  addClientCard()        ->  Insert client card DOM
client_disconnected      ->  removeClientCard()     ->  Fade out client card
round_started            ->  onRoundStart()         ->  Update phase indicator
                                                        Enable stop button
                                                        Update progress bar
update_received          ->  onUpdateReceived()     ->  Flash client card border
                                                        Update client's round count
round_completed          ->  onRoundComplete()      ->  Push data to all charts
                                                        Update progress bar
                                                        Update leaderboard table
                                                        Update comparison chart
                                                        Animate F1 bars
training_complete        ->  onTrainingComplete()   ->  Set phase to "Complete"
                                                        Enable export buttons
                                                        Show final summary
```

### 5.3 Chart.js Initialization

All charts initialized on `DOMContentLoaded`. Canvases have fixed aspect ratios via CSS `aspect-ratio: 16/9`. Charts use the warm color tokens via JS constants (CSS custom properties read once at init).

### 5.4 Initial Data Load

On page load (before WebSocket connects), fetch current state via REST:
```javascript
async loadInitialState() {
  const [status, metrics, leaderboard, comparison] = await Promise.all([
    fetch('/api/training/status').then(r => r.json()),
    fetch('/api/metrics/history').then(r => r.json()),
    fetch('/api/metrics/leaderboard').then(r => r.json()),
    fetch('/api/metrics/comparison').then(r => r.json()),
  ]);
  
  this.renderStatus(status);
  this.renderMetricsHistory(metrics);   // Backfill charts with historical data
  this.renderLeaderboard(leaderboard);
  this.renderComparison(comparison);
}
```

This ensures the dashboard is fully populated even if opened mid-session.

---

## 6. Micro-Animations Inventory

| Element | Trigger | Animation |
|:---|:---|:---|
| NM-Button | `:active` | Shadow transitions to `--nm-pressed`, scale 0.98, 200ms |
| NM-Button | `:hover` | Shadow strengthens, scale 1.01, 200ms |
| Status Dot (online) | Continuous | Pulse: scale 1->1.4, opacity 1->0, 2s infinite |
| Status Dot (training) | Continuous | Pulse: faster (1s), `--accent-gold` |
| Client Card | Connect | `slideIn`: translateY(20px)->0, opacity 0->1, 300ms ease-out |
| Client Card | Disconnect | `fadeOut`: opacity 1->0.5, 300ms ease-in |
| Client Card | Update received | Border flash: `--accent-gold` border, 400ms fade |
| Progress Bar Fill | Round complete | Width transition, 600ms ease-out |
| Chart Data Point | New data | Chart.js easeOutQuart, 500ms |
| Phase Indicator | State change | Color crossfade, 300ms |
| Sidebar Nav Item | `:hover` | Background tint fade, 200ms |
| Sidebar Nav Item | Active | Left border accent, `--nm-pressed` background |
| Modal | Open | Overlay fade 200ms, modal scale 0.95->1 + fade 200ms |
| Toggle Thumb | State change | `translateX` 200ms ease |
| Leaderboard Row | Rank change | Subtle background flash 400ms |
| Export Button | Download start | Brief scale pulse (1->1.05->1), 300ms |

---

## 7. Sidebar Navigation

**Items (top to bottom):**

| Icon (SVG) | Label | Scrolls To |
|:---|:---|:---|
| Grid/Dashboard | Overview | Top of page |
| Users | Clients | `#panel-clients` |
| Play/Cog | Training | `#panel-training` |
| Chart Line | Metrics | `#panel-accuracy` |
| Bar Chart | Comparison | `#panel-comparison` |
| Trophy | Leaderboard | `#panel-leaderboard` |
| Shield | Privacy | `#panel-privacy` |
| Download | Export | `#panel-export` |

**Behavior:** Clicking a nav item smooth-scrolls the main content area to the target panel. Active item determined by `IntersectionObserver` on panels.

**Bottom of sidebar:**
- Server connection status dot
- "FGT v1.0" version text in `--text-muted`

---

## 8. Accessibility

| Concern | Implementation |
|:---|:---|
| **Contrast** | All text meets WCAG AA (4.5:1) against `--bg-base`. Verified: `--text-primary` (#3D2B1F) on `--bg-base` (#F0E4D7) = ~8.2:1. |
| **Keyboard Navigation** | All buttons, inputs, toggles are focusable. Focus ring: 2px solid `--accent-primary` with 2px offset. |
| **ARIA** | Toggles have `role="switch"` + `aria-checked`. Progress bars have `role="progressbar"` + `aria-valuenow`. Status dots have `aria-label`. Charts have `aria-label` descriptions. |
| **Screen Reader** | Live region (`aria-live="polite"`) for training status updates. Client connect/disconnect announced. |
| **Reduced Motion** | `@media (prefers-reduced-motion: reduce)` disables pulse animations, sets all transitions to 0ms. |

---

## 9. HTML Structure Outline

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FGT Dashboard - Federated Gallery Tags</title>
  <meta name="description" content="Real-time monitoring dashboard for Federated Gallery Tags training">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400&family=Outfit:wght@400;500&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="/dashboard/styles.css">
</head>
<body>
  <div class="dashboard" id="dashboard">
    
    <!-- Sidebar -->
    <aside class="sidebar" id="sidebar">
      <div class="sidebar__logo"><!-- Title --></div>
      <nav class="sidebar__nav" aria-label="Dashboard navigation">
        <!-- Nav items -->
      </nav>
      <div class="sidebar__footer"><!-- Status + version --></div>
    </aside>
    
    <!-- Header -->
    <header class="header" id="header">
      <h1 class="header__title">Federated Gallery Tags</h1>
      <div class="header__meta">
        <!-- IP, model version, connection status -->
      </div>
    </header>
    
    <!-- Main Content -->
    <main class="main-content" id="main-content">
      <section class="nm-card" id="panel-status"><!-- Status --></section>
      <section class="nm-card" id="panel-clients"><!-- Clients --></section>
      <section class="nm-card nm-card--full" id="panel-training"><!-- Controls --></section>
      <section class="nm-card" id="panel-accuracy"><!-- Chart --></section>
      <section class="nm-card" id="panel-loss"><!-- Chart --></section>
      <section class="nm-card nm-card--full" id="panel-f1"><!-- F1 Chart --></section>
      <section class="nm-card" id="panel-comparison"><!-- Comparison --></section>
      <section class="nm-card" id="panel-leaderboard"><!-- Leaderboard --></section>
      <section class="nm-card" id="panel-privacy"><!-- Privacy --></section>
      <section class="nm-card" id="panel-export"><!-- Export --></section>
    </main>
    
  </div>
  
  <!-- Modals -->
  <div class="nm-modal-overlay" id="modal-export" hidden><!-- ... --></div>
  
  <!-- Live Region for Screen Readers -->
  <div class="sr-only" aria-live="polite" id="live-announcer"></div>
  
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <script src="/dashboard/dashboard.js"></script>
</body>
</html>
```

---

## 10. File Size Targets

| File | Target | Notes |
|:---|:---|:---|
| `index.html` | < 8 KB | Semantic markup, no inline styles |
| `styles.css` | < 12 KB | Full design system + all components |
| `dashboard.js` | < 15 KB | All logic, no framework overhead |
| **Total (excl. Chart.js CDN)** | **< 35 KB** | Instant load on LAN |
| Chart.js (CDN) | ~65 KB gzipped | Cached after first load |

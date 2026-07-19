# GalleryFL — Design System (Blush / Web3 Soft)

A single visual language shared by the **Android app** and the **web Coordinator
console**, so the two surfaces read as one product. "Fun but serious": a calm
photos-first experience on Android, a precise operational console on the web.

## 1. Product personality
- **Photos-first, ML-second** on Android; **state/control-first** on the dashboard.
- Advanced FL / privacy / explainability features are progressively disclosed,
  never forced into the primary surface.
- Tone: warm, confident, privacy-aware. Microcopy is plain and human.

## 2. Palette (single source of truth)
Applied identically in `android/.../ui/theme/Color.kt` and
`server/dashboard/styles.css`.

| Token | Hex | Use |
|---|---|---|
| BgBase | `#FDF6FB` | App / page background (near-white pink) |
| BgSurface | `#FFFFFF` | Cards, sheets |
| BgGlass | `rgba(239,77,155,0.15)` | Glass / overlay tint |
| Primary | `#EF4D9B` | Rose — primary actions, active states |
| Secondary | `#8B7FE8` | Lavender — secondary accents, charts |
| Amber | `#F5A623` | Privacy / DP emphasis |
| Success | `#16B87A` | Online, healthy |
| Error | `#F0445E` | Destructive, failures |
| TextPrimary | `#2A1B2E` | Ink plum |
| TextSecondary | `rgba(42,27,46,0.62)` | Supporting text |

## 3. Typography
- Sans: **Outfit** (headings/UI) / **Inter** (body) on web; Android uses the
  system font with bold weights for emphasis.
- Mono: **JetBrains Mono** for codes, metrics, parameters.
- Scale: large image-first titles on Android; compact, dense labels on the
  console.

## 4. Shape & depth
- Radii: sm 10 · md 16 · lg 22 · xl 30 · pill 999.
- Soft-depth shadows, pink-tinted, layered (`--shadow-sm/md/lg`). No harsh
  drop shadows. Generous whitespace.

## 5. Components (shared vocabulary)
- Cards / surfaces, pill badges, soft buttons (primary = rose→lavender gradient),
  inset inputs, pill progress, client/leaderboard rows, toast, modal.
- Charts: self-contained canvas (no CDN) — line (accuracy/loss), grouped bars
  (comparison), horizontal bars (per-class F1).

## 6. Naming consistency
| Concept | Android | Dashboard |
|---|---|---|
| Server | Coordinator | Coordinator |
| FL rounds | Model Sync | Training |
| Tag system | Tags | Taxonomy |
| Privacy | Privacy | Privacy (DP ε/δ/C) |
| Organize | Organize / Smart Albums | — |

## 7. Usage rules
- Images/content first on Android; state/control first on the dashboard.
- One primary action per surface; secondary actions in sheets/dialogs.
- Never show raw JSON or "programmer text" in empty/error states — use
  intentional, on-brand copy.
- Accessibility: visible focus, status dots, `aria-live` for phase changes.

# MeetAssist design system (source of truth: `meetassist/ui/theme.py`)

Direction (ui-ux-pro-max search, "meeting recorder transcription productivity desktop minimal"): **Minimalism / Swiss, Apple-style** -
"recording red + waveform blue". Native Windows 11 feel: Segoe UI Variable (SF equivalent), Nirmala UI for Tamil, DWM caption colour = sidebar.

## Layers
primitives (Palette values) -> semantic roles (bg, sidebar, surface, fill, text/text2/text3, sep, accent, accent_fill, record, success, warning, you, remote)
-> components (`build_qss`, `widgets.py`). No colour is hard-coded in a widget; both themes are separate palettes (never derived), light and dark.

## Tokens
- Spacing (4-pt): 2/4/8/12/16/24/32/48.  Radius: 6/10/14/20.  Icons: 14/18/22/28 (Lucide-style, 1.75 stroke, one family, no emoji).
- Type (px): caption 12, footnote 13, body 14, callout 15, headline 17, timer 22 (tabular digits), title 26, hero 30. Reading text 16 px / 155 % leading, 760 px measure.
- Motion: fast 120 / base 200 / slow 320 ms, pulse 1.7 s. Record button morph disc->stop square; segmented control slides; toast fades; switches slide.
- Contrast (enforced by tests/test_ui.py, WCAG): text/text2 >= 7:1, text3 >= 4.5:1 on bg, sidebar and fills; accent text >= 4.5:1; on_accent on accent_fill >= 4.5:1; both themes.
  (The test caught two real defects: Remote-green 4.21 and white-on-bright-blue dark button 3.0 -> fixed with darker green and `accent_fill`.)

## Layout
Sidebar 300 px (brand, search, meetings grouped Today/Yesterday/Previous 7 days/Earlier, Import, Settings) | content: title + status pill, banner (warn/error), toolbar
(segmented Transcript/Cleaned/Summary; Tamil-view menu, copy, save, folder, AI menu, more) | reading column | **floating recorder dock** (record button, timer/title, live two-lane
waveform: You blue up / Remote green down, level meters). Dock sits in layout (never overlaps content).

## Interaction / accessibility
Visible 2 px focus rings on every control, Tab order = visual order, accessible names + tooltips on all icon buttons, shortcuts: Ctrl+R record, Ctrl+O import, Ctrl+F search,
Ctrl+S save, Ctrl+Shift+C copy, Ctrl+, settings. Follows Windows light/dark live; override in Settings > Appearance. Empty states explain next action. Warnings are inline banners,
confirmations are toasts, destructive actions confirm.

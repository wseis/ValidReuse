"""Shared look and feel for the ValidReuse workspaces.

Keeping the stylesheet and the card/hero primitives in one module means the
count workflow and the documentation render as the same application rather than
two separately styled pages.
"""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(143, 179, 255, 0.16), transparent 30%),
        radial-gradient(circle at 88% 0%, rgba(67, 97, 238, 0.14), transparent 26%),
        rgb(246, 246, 254);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 78rem; }

.vr-hero {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 26px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 20px 50px rgba(148, 163, 184, 0.18);
    margin-bottom: 1.1rem;
}
.vr-hero h1 { margin: 0; font-size: 2rem; letter-spacing: -0.02em; color: #132238; }
.vr-hero p { margin: 0.6rem 0 0; color: #475569; font-size: 1rem; max-width: 62rem; }

.vr-stepper { display: flex; align-items: center; gap: 0.35rem; margin: 0 0 1.2rem; flex-wrap: wrap; }
.vr-step {
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0.42rem 0.85rem; border-radius: 999px;
    background: rgba(255, 255, 255, 0.7);
    border: 1px solid rgba(15, 23, 42, 0.06);
    color: #94a3b8; font-size: 0.85rem; font-weight: 600; white-space: nowrap;
}
.vr-step .vr-dot {
    width: 1.35rem; height: 1.35rem; border-radius: 50%;
    display: grid; place-items: center; font-size: 0.75rem;
    background: #e2e8f0; color: #94a3b8;
}
.vr-step.current { color: #132238; border-color: rgba(37, 99, 235, 0.35); background: rgba(219, 234, 254, 0.65); }
.vr-step.current .vr-dot { background: #2563eb; color: #ffffff; }
.vr-step.done { color: #132238; }
.vr-step.done .vr-dot { background: #16a34a; color: #ffffff; }
.vr-step-sep { flex: 1 1 1rem; min-width: 0.7rem; height: 2px; background: rgba(15, 23, 42, 0.08); border-radius: 2px; }

.vr-h { display: flex; align-items: center; gap: 0.6rem; margin: 0 0 0.35rem; }
.vr-h .vr-num {
    width: 1.7rem; height: 1.7rem; border-radius: 9px; flex: none;
    display: grid; place-items: center;
    background: #2563eb; color: #ffffff; font-weight: 700; font-size: 0.9rem;
}
.vr-h .vr-num.vr-wide { width: auto; padding: 0 0.5rem; font-size: 0.8rem; letter-spacing: 0.03em; }
.vr-h h3 { margin: 0; font-size: 1.15rem; color: #132238; font-weight: 650; }
.vr-h.locked .vr-num { background: #cbd5e1; color: #ffffff; }
.vr-h.locked h3 { color: #94a3b8; }

/* st.container(border=True, key="vr_...") renders with a matching .st-key-vr_... class. */
[class*="st-key-vr_"] {
    border-radius: 22px;
    border-color: rgba(15, 23, 42, 0.07);
    background: rgba(255, 255, 255, 0.78);
    box-shadow: 0 16px 38px rgba(148, 163, 184, 0.10);
    padding: 1.15rem 1.35rem;
    margin-bottom: 0.9rem;
}
[data-testid="stExpander"] { border-radius: 14px; border-color: rgba(37, 99, 235, 0.12); }
[data-testid="stMetric"] { background: rgba(251, 252, 255, 0.9); border-radius: 14px; }
/* Ranges such as "1875-3750 mL" do not fit at the default metric size. */
[data-testid="stMetricValue"] { font-size: 1.65rem; }

.vr-side { font-size: 0.84rem; color: #475569; }
.vr-side .vr-row {
    display: flex; justify-content: space-between; gap: 0.6rem;
    padding: 0.28rem 0; border-bottom: 1px dashed rgba(15, 23, 42, 0.10);
}
.vr-side .vr-row:last-child { border-bottom: none; }
.vr-side .vr-row b { color: #132238; font-variant-numeric: tabular-nums; text-align: right; }
.vr-side .vr-ok { color: #16a34a; }
.vr-side .vr-bad { color: #dc2626; }
.vr-side .vr-idle { color: #94a3b8; }

</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title, lead):
    st.markdown(
        f'<div class="vr-hero"><h1>{title}</h1><p>{lead}</p></div>',
        unsafe_allow_html=True,
    )


def section_head(badge, title, locked=False):
    """Card heading with a numbered (or short-text) badge."""
    badge = str(badge)
    badge_class = "vr-num vr-wide" if len(badge) > 2 else "vr-num"
    css_class = "vr-h locked" if locked else "vr-h"
    st.markdown(
        f'<div class="{css_class}"><span class="{badge_class}">{badge}</span><h3>{title}</h3></div>',
        unsafe_allow_html=True,
    )


def note(markdown):
    """Short aside. Uses st.info so markdown and $...$ math are rendered."""
    st.info(markdown)

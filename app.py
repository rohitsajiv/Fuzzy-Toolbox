import streamlit as st
import numpy as np
import json, io
import plotly.graph_objects as go
from toolbox import (
    build_membership, equal, complement, union, intersection,
    alg_sum, alg_prod, bsum, bdiff, alg_diff, power, crisp_mul,
    DEFUZZ, fire_rule, aggregate
)

st.set_page_config(page_title="Fuzzy Toolbox • Tabs", layout="wide", page_icon="🧩")

# ---- session ----
if "page" not in st.session_state: st.session_state.page = 0
if "universes" not in st.session_state: st.session_state.universes = {}
if "sets" not in st.session_state: st.session_state.sets = {}
if "rules" not in st.session_state: st.session_state.rules = []
if "last_outputs" not in st.session_state: st.session_state.last_outputs = {}

PAGES = ["🏗 Universes", "🌈 Fuzzy Sets", "⚖ Rules", "🧠 Inference", "📊 Results"]

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def rerun_safe():
    """Safe rerun compatible with Streamlit ≥1.31"""
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

def ensure_sets_universe(u):
    if u not in st.session_state.sets:
        st.session_state.sets[u] = {}

def plot_universe(u, extra=None, title=""):
    U = st.session_state.universes[u]
    fig = go.Figure()
    for s, d in st.session_state.sets.get(u, {}).items():
        fig.add_trace(go.Scatter(x=U, y=d["mu"], mode="lines+markers", name=s))
    if extra:
        for label, mu in extra:
            fig.add_trace(go.Scatter(x=U, y=mu, mode="lines", name=label, line=dict(width=3, dash="dash")))
    fig.update_layout(
        template="plotly_dark", title=f"{u} {title}".strip(),
        xaxis_title="Universe", yaxis_title="Membership", height=360
    )
    st.plotly_chart(fig, use_container_width=True)

def df_values(u, label, mu):
    st.dataframe({"U": st.session_state.universes[u], label: np.round(mu, 4)}, use_container_width=True)

# ---- top nav ----
st.markdown("<h1 style='margin:0'>🧩 Fuzzy Logic Toolbox</h1>", unsafe_allow_html=True)
sel = st.radio("Navigate", PAGES, index=st.session_state.page, horizontal=True, label_visibility="collapsed")
st.session_state.page = PAGES.index(sel)
st.write("---")

# =========================================================
# 1) UNIVERSES
# =========================================================
if st.session_state.page == 0:
    st.subheader("Define Universes")

    c1, c2 = st.columns([2, 3])
    with c1:
        u_name = st.text_input("Universe name")
        u_vals = st.text_input("Values (space-separated, e.g. 0 10 20 ... 100)")
        col = st.columns(2)
        if col[0].button("Add universe"):
            try:
                U = list(map(float, u_vals.split()))
                assert len(U) >= 2
                st.session_state.universes[u_name] = U
                rerun_safe()
            except Exception:
                st.error("Enter ≥2 numeric values")
        if col[1].button("Clear all"):
            st.session_state.universes.clear()
            st.session_state.sets.clear()
            st.session_state.rules.clear()
            st.session_state.last_outputs.clear()
            rerun_safe()

    with c2:
        st.markdown("### Existing Universes")
        if not st.session_state.universes:
            st.info("None yet.")
        for u, U in list(st.session_state.universes.items()):
            b1, b2, b3 = st.columns([5, 1, 1])
            b1.write(f"**{u}**: {U}")
            if b2.button("Edit", key=f"edit_u_{u}"):
                new = st.text_input(f"Edit values for {u}", value=" ".join(map(str, U)), key=f"in_u_{u}")
                if st.button("Save", key=f"save_u_{u}"):
                    try:
                        newU = list(map(float, new.split()))
                        st.session_state.universes[u] = newU
                        # recompute sets on this universe
                        for s, d in st.session_state.sets.get(u, {}).items():
                            if d["type"] == "Manual":
                                if len(d["mu"]) != len(newU):
                                    d["mu"] = np.clip(np.interp(newU, U, d["mu"]), 0, 1)
                            else:
                                d["mu"] = build_membership(newU, d["type"], d["params"], None)
                        rerun_safe()
                    except Exception:
                        st.error("Invalid numbers")
            if b3.button("Delete", key=f"del_u_{u}"):
                st.session_state.universes.pop(u, None)
                st.session_state.sets.pop(u, None)
                rerun_safe()

    if st.button("Next → Fuzzy Sets"):
        st.session_state.page = 1
        rerun_safe()

# =========================================================
# 2) FUZZY SETS
# =========================================================
elif st.session_state.page == 1:
    if not st.session_state.universes:
        st.warning("Add a universe first.")
        if st.button("← Back"):
            st.session_state.page = 0
            rerun_safe()
    else:
        st.subheader("Create / Edit Fuzzy Sets")
        u_sel = st.selectbox("Universe", list(st.session_state.universes.keys()))
        ensure_sets_universe(u_sel)
        U = st.session_state.universes[u_sel]

        c1, c2 = st.columns([2, 2])
        with c1:
            s_name = st.text_input("Set name")
            mtype = st.selectbox("Membership type", ["Triangular", "Trapezoidal", "Gaussian", "Manual"])
            params, manual = None, None

            if mtype == "Triangular":
                a = st.number_input("a", value=float(U[0]))
                b = st.number_input("b", value=float(U[len(U)//2]))
                c = st.number_input("c", value=float(U[-1]))
                params = [a, b, c]
            elif mtype == "Trapezoidal":
                a = st.number_input("a"); b = st.number_input("b")
                c = st.number_input("c"); d = st.number_input("d")
                params = [a, b, c, d]
            elif mtype == "Gaussian":
                c0 = st.number_input("center", value=float(U[len(U)//2]))
                s0 = st.number_input("sigma", min_value=0.001, value=10.0, step=0.5)
                params = [c0, s0]
            else:
                man = st.text_input(f"Manual μ values (need {len(U)})")
                if man:
                    try:
                        manual = list(map(float, man.split()))
                    except Exception:
                        st.warning("Enter numeric values")

            cols = st.columns(2)
            if cols[0].button("Preview"):
                try:
                    mu = build_membership(U, mtype, params, manual)
                    plot_universe(u_sel, extra=[(f"preview:{s_name or mtype}", mu)], title="(preview)")
                    df_values(u_sel, "preview", mu)
                except Exception as e:
                    st.error(str(e))
            if cols[1].button("Add / Update"):
                try:
                    mu = build_membership(U, mtype, params, manual)
                    st.session_state.sets[u_sel][s_name] = {"type": mtype, "params": params, "mu": mu}
                    st.success(f"Saved set '{s_name}' in '{u_sel}'")
                    rerun_safe()
                except Exception as e:
                    st.error(str(e))

        with c2:
            st.markdown("### Sets in this universe")
            if not st.session_state.sets[u_sel]:
                st.info("None.")
            for s, d in list(st.session_state.sets[u_sel].items()):
                b1, b2, b3 = st.columns([5, 1, 1])
                b1.write(f"**{s}** · {d['type']} · params={d['params']}")
                if b2.button("Edit", key=f"edit_s_{u_sel}_{s}"):
                    plot_universe(u_sel, extra=[(f"editing:{s}", d["mu"])], title="(editing)")
                    df_values(u_sel, f"µ({s})", d["mu"])
                if b3.button("Delete", key=f"del_s_{u_sel}_{s}"):
                    st.session_state.sets[u_sel].pop(s, None)
                    rerun_safe()

        if st.button("Next → Rules"):
            st.session_state.page = 2
            rerun_safe()

# =========================================================
# 3) RULES
# =========================================================
elif st.session_state.page == 2:
    st.subheader("Rule Base")
    all_unis = list(st.session_state.universes.keys())
    if not all_unis:
        st.warning("Add universes and sets first.")
        if st.button("← Back"):
            st.session_state.page = 1
            rerun_safe()
    else:
        if st.button("Add rule"):
            st.session_state.rules.append({"antecedents": [], "op": "AND", "consequent": None, "implication": "Mamdani"})
            rerun_safe()

        for i, rule in enumerate(st.session_state.rules):
            st.markdown(f"**Rule {i+1}**")
            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button("Add antecedent", key=f"addant_{i}"):
                for u in all_unis:
                    if st.session_state.sets.get(u):
                        rule["antecedents"].append((u, next(iter(st.session_state.sets[u].keys()))))
                        rerun_safe()
                        break
            rule["op"] = c2.selectbox("Operator", ["AND", "OR"], index=["AND", "OR"].index(rule["op"]), key=f"op_{i}")
            if c3.button("Delete rule", key=f"delrule_{i}"):
                st.session_state.rules.pop(i)
                rerun_safe()

            for j, (u, s) in enumerate(rule["antecedents"]):
                a1, a2, a3 = st.columns([1.2, 1.2, 0.4])
                nu = a1.selectbox("Universe", all_unis, index=all_unis.index(u), key=f"ru_{i}_{j}")
                sets_here = list(st.session_state.sets.get(nu, {}).keys())
                ns = a2.selectbox("Set", sets_here, index=sets_here.index(s) if s in sets_here else 0, key=f"rs_{i}_{j}")
                if a3.button("✖", key=f"rmant_{i}_{j}"):
                    rule["antecedents"].pop(j)
                    rerun_safe()
                rule["antecedents"][j] = (nu, ns)

            b1, b2, b3 = st.columns([1.2, 1.2, 1])
            rule["implication"] = b1.selectbox("Implication", ["Mamdani", "Larsen", "Zadeh"], key=f"imp_{i}")
            cu = b2.selectbox("Consequent universe", all_unis, key=f"cu_{i}")
            csets = list(st.session_state.sets.get(cu, {}).keys())
            cs = b3.selectbox("Consequent set", csets, key=f"cs_{i}")
            rule["consequent"] = (cu, cs)

        if st.button("Next → Inference"):
            st.session_state.page = 3
            rerun_safe()

# =========================================================
# 4) INFERENCE
# =========================================================
elif st.session_state.page == 3:
    st.subheader("Inference")
    if not st.session_state.rules:
        st.info("Add at least one rule first.")
        if st.button("← Back"):
            st.session_state.page = 2
            rerun_safe()
    else:
        crisp = {}
        for u, U in st.session_state.universes.items():
            step = float(U[1] - U[0]) if len(U) > 1 else 1.0
            crisp[u] = st.slider(f"Crisp input for {u}", float(min(U)), float(max(U)), float(U[len(U)//2]), step=step)

        outputs, alphas, labels = [], [], []
        for idx, rule in enumerate(st.session_state.rules):
            if not rule["antecedents"] or not rule["consequent"] or not rule["consequent"][1]:
                continue
            cu, Uc, mu_out, a = fire_rule(rule, crisp, st.session_state.universes, st.session_state.sets)
            outputs.append((cu, Uc, mu_out))
            alphas.append(a)
            labels.append(f"R{idx+1}-{rule['implication']}->{rule['consequent'][1]}")
            st.caption(f"Rule output on **{cu}** (α={a:.3f})")
            st.dataframe({"U": Uc, "μ_rule": np.round(mu_out, 4)}, use_container_width=True)

        if outputs:
            agg = aggregate(outputs)
            st.session_state.last_outputs = agg

            fig = go.Figure(go.Bar(x=labels, y=alphas))
            fig.update_layout(template="plotly_dark", height=260, title="Firing strengths (α)")
            st.plotly_chart(fig, use_container_width=True)

            for u, (U, mu) in agg.items():
                fig = go.Figure()
                for s, d in st.session_state.sets.get(u, {}).items():
                    fig.add_trace(go.Scatter(x=U, y=d["mu"], mode="lines", name=s, opacity=0.35))
                fig.add_trace(go.Scatter(x=U, y=mu, mode="lines", name="Aggregated", line=dict(width=3)))
                fig.update_layout(template="plotly_dark", height=360, title=f"Aggregated on {u}")
                st.plotly_chart(fig, use_container_width=True)

        if st.button("Next → Results"):
            st.session_state.page = 4
            rerun_safe()

# =========================================================
# 5) RESULTS
# =========================================================
else:
    st.subheader("Results & Defuzzification")
    if not st.session_state.last_outputs:
        st.info("Run inference first.")
        if st.button("← Back"):
            st.session_state.page = 3
            rerun_safe()
    else:
        for u, (U, mu) in st.session_state.last_outputs.items():
            st.markdown(f"### Output universe: **{u}**")
            method = st.selectbox(f"Defuzzification method ({u})", list(DEFUZZ.keys()), key=f"def_{u}")
            val = DEFUZZ[method](np.array(U, float), np.array(mu, float))

            fig = go.Figure(go.Scatter(x=U, y=mu, mode="lines", name="Aggregated", line=dict(width=3)))
            if val == val:
                fig.add_vline(x=val, line=dict(color="red", dash="dash"), annotation_text=f"{method}: {val:.3f}")
            fig.update_layout(template="plotly_dark", height=360)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe({"U": U, "μ_aggregated": np.round(mu, 4)}, use_container_width=True)
            if val == val:
                st.success(f"Defuzzified value = {val:.3f}")
            else:
                st.warning("Cannot defuzzify (zero area).")

        if st.button("← Back"):
            st.session_state.page = 3
            rerun_safe()
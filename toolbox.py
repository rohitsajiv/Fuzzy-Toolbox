import numpy as np

# --- membership ---
def _tri(U, a, b, c):
    U = np.asarray(U, float); mu = np.zeros_like(U)
    if a == b:
        mu[U <= b] = 1.0
        m = (U > b) & (U <= c)
        if c > b: mu[m] = (c - U[m]) / (c - b)
    elif b == c:
        mu[U >= b] = 1.0
        m = (U >= a) & (U < b)
        if b > a: mu[m] = (U[m] - a) / (b - a)
    else:
        m = (U >= a) & (U <= b)
        if b > a: mu[m] = (U[m] - a) / (b - a)
        m = (U >= b) & (U <= c)
        if c > b: mu[m] = (c - U[m]) / (c - b)
    return np.clip(mu, 0, 1)

def _trap(U, a, b, c, d):
    U = np.asarray(U, float); mu = np.zeros_like(U)
    if a == b:
        mu[U <= b] = 1.0
        mu[(U >= b) & (U <= c)] = 1.0
        m = (U >= c) & (U <= d)
        if d > c: mu[m] = (d - U[m]) / (d - c)
        return mu
    if c == d:
        mu[U >= c] = 1.0
        m = (U >= a) & (U <= b)
        if b > a: mu[m] = (U[m] - a) / (b - a)
        mu[(U >= b) & (U <= c)] = 1.0
        return mu
    m = (U >= a) & (U <= b)
    if b > a: mu[m] = (U[m] - a) / (b - a)
    mu[(U >= b) & (U <= c)] = 1.0
    m = (U >= c) & (U <= d)
    if d > c: mu[m] = (d - U[m]) / (d - c)
    return np.clip(mu, 0, 1)

def _gauss(U, c, s):
    U = np.asarray(U, float); s = max(float(s), 1e-9)
    return np.exp(-((U - c) ** 2) / (2 * s ** 2))

def build_membership(U, mtype, params=None, manual=None):
    if mtype == "Triangular":
        a, b, c = params;  return _tri(U, a, b, c)
    if mtype == "Trapezoidal":
        a, b, c, d = params; return _trap(U, a, b, c, d)
    if mtype == "Gaussian":
        c0, s0 = params;     return _gauss(U, c0, s0)
    if mtype == "Manual":
        vals = np.asarray(manual, float)
        if len(vals) != len(U): raise ValueError("Manual values length must match universe.")
        return np.clip(vals, 0, 1)
    raise ValueError("Unknown membership type.")

# --- set ops (same universe) ---
def equal(muA, muB):            return bool(np.allclose(muA, muB))
def complement(muA):            return 1.0 - muA
def union(muA, muB):            return np.maximum(muA, muB)
def intersection(muA, muB):     return np.minimum(muA, muB)
def alg_sum(muA, muB):          return np.clip(muA + muB - muA * muB, 0, 1)
def alg_prod(muA, muB):         return np.clip(muA * muB, 0, 1)
def bsum(muA, muB):             return np.clip(muA + muB, 0, 1)
def bdiff(muA, muB):            return np.clip(muA - muB, 0, 1)
def alg_diff(muA, muB):         return np.clip(muA - muB, 0, 1)
def power(muA, k):              return np.clip(muA ** float(k), 0, 1)
def crisp_mul(muA, k):          return np.clip(muA * float(k), 0, 1)

# --- implication (scalar α on consequent μB) ---
def imply(alpha, muB, mode):
    if mode == "Mamdani": return np.minimum(alpha, muB)
    if mode == "Larsen":  return np.clip(alpha * muB, 0, 1)
    if mode == "Zadeh":   return np.maximum(1 - alpha, np.minimum(alpha, muB))
    raise ValueError("Unknown implication.")

# --- defuzz ---
def _centroid(U, mu):
    s = np.sum(mu);  return np.nan if s <= 1e-12 else float(np.sum(U * mu) / s)
def _bisector(U, mu):
    cum = np.cumsum(mu); tot = cum[-1]
    if tot <= 1e-12: return np.nan
    i = int(np.searchsorted(cum, tot/2)); i = min(i, len(U)-1); return float(U[i])
def _mom(U, mu): m = np.max(mu); idx = np.where(mu == m)[0]; return float(np.mean(U[idx])) if idx.size else np.nan
def _som(U, mu): m = np.max(mu); idx = np.where(mu == m)[0]; return float(U[idx[0]]) if idx.size else np.nan
def _lom(U, mu): m = np.max(mu); idx = np.where(mu == m)[0]; return float(U[idx[-1]]) if idx.size else np.nan
def _lcut(U, mu, lam=0.5): idx = np.where(mu >= lam)[0]; return float(np.mean(U[idx])) if idx.size else np.nan
def _height(U, mu): idx = np.where(mu > 0)[0]; return float(np.sum(U[idx]*mu[idx])/np.sum(mu[idx])) if idx.size else np.nan
def _cos(U, mu): idx = np.where(mu > 0)[0]; return float((np.min(U[idx])+np.max(U[idx]))/2) if idx.size else np.nan

DEFUZZ = {
    "Centroid / COG / COA": _centroid,
    "Bisector of Area (BOA)": _bisector,
    "Mean of Maximum (MOM)": _mom,
    "Smallest of Maximum (SOM)": _som,
    "Largest of Maximum (LOM)": _lom,
    "Lambda-cut Method (λ=0.5)": lambda U, mu: _lcut(U, mu, 0.5),
    "Weighted Average (same as centroid)": _centroid,
    "Height Method": _height,
    "Center of Sums (COS)": _cos,
}

# --- rules ---
def _interp_mu(x, U, mu):
    U = np.asarray(U, float); mu = np.asarray(mu, float)
    return float(np.interp(x, U, mu))

# rule: {"antecedents":[(u,set),...], "op":"AND"/"OR", "consequent":(u,set), "implication":"Mamdani"/"Larsen"/"Zadeh"}
def fire_rule(rule, crisp_inputs, universes, sets):
    degs = []
    for u, s in rule["antecedents"]:
        U = universes[u]; mu = sets[u][s]["mu"]; x = float(crisp_inputs[u])
        degs.append(_interp_mu(x, U, mu))
    alpha = min(degs) if rule["op"] == "AND" else max(degs)
    cu, cs = rule["consequent"]
    Uc = universes[cu]; muB = sets[cu][cs]["mu"]
    return cu, Uc, imply(alpha, muB, rule["implication"]), alpha

def aggregate(outputs):
    by_u = {}
    for u, U, mu in outputs:
        if u not in by_u: by_u[u] = (U, mu.copy())
        else:
            U0, agg = by_u[u]; by_u[u] = (U0, np.maximum(agg, mu))
    return by_u

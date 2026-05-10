# §IV (Results and Discussion) — Restructure Plan

Date: 2026-05-07
Status: agreed structure; B2 in progress.

## 1. New 4-section layout

| New | Content focus |
|---|---|
| §IV.A — Realized attractors and their spatiotemporal structure | Phase diagram + 3 default-IC attractors (cold-swollen, LCST-front, frozen-front) |
| §IV.B — Relaxation oscillator on a spatially-generated slow manifold | Quantitative mechanism: fast/slow split, slow-manifold geometry, 4-phase cycle, period scaling, self-limiting collapse |
| §IV.C — Linear stability is necessary but not sufficient | Whitney-cusp homogeneous-SS existence map (analytic) + SNIC period-divergence scaling; "necessary-but-not-sufficient" stated as a corollary of §IV.B's slow manifold (no LSA-vs-PDE map) |
| §IV.D — A hidden hot-runaway state and the IC-induced hysteresis prediction | C-cell IC sweep, refined basins, hot-runaway mechanism, quantified hysteresis loop |

§IV opens with a one-paragraph takeaway listing the four findings.

## 2. New analyses (priority ★ = evidentiary value / cost)

| ID | Task | Section | ★ |
|---|---|---|---|
| B1 | Fast/slow timescale numbers (4 phases) | §IV.B | ★★ |
| B2 | Slow-manifold + limit cycle in (θ, J) | §IV.B | ★★★★★ |
| B3 | Period scaling T ≈ τ_quench | §IV.B | ★★★★ |
| C1 | Homogeneous SS real-root existence map (analytic) | §IV.C | ★★★★ |
| C2 | T_PDE / T_LSA contrast | (RETIRED — coarse PDE grid; subsumed by §IV.B slow-manifold argument) | — |
| C3 | SNIC period-divergence fit | §IV.C | ★★ |
| D1 | Refined basin boundary at C-cell | §IV.D | ★★★ |
| D2 | Quantified IC-hysteresis curve | §IV.D | ★★★★★ |
| D3 | Hot-runaway thin-flame verification (SI) | §IV.D / SI | ★ |

## 3. Figure layout (after restructure)

| Figure | Panels | Notes |
|---|---|---|
| Fig.2 | (a) regime (Bi_T,S_χ); (b) regime (Bi_T,Da); (c) period; (d) peak φ | ✅ 4 panels (amplitude, mean-J(ξ) dropped) |
| Fig.3 | 3 rows: cold-swollen / LCST-front / frozen-front; J(ξ,t) \| θ(ξ,t) \| J_surf+J_centre(t) | ✅ 3 rows (hot-runaway moved to Fig.6) |
| Fig.4 | (a) schematic; (b) time series; (c) **B2 slow-manifold + limit cycle**; (d) D(φ) heatmap with ξ_LCST line; (e) **B3 period scaling** | ✅ 5 panels (J/θ/u kymographs collapsed into D(φ); B2/B3 absorbed) |
| Fig.5 | (a) **C1 homogeneous-SS + Whitney-cusp fold**; (b) **C3 SNIC period scaling** | ✅ 2-panel composite figure |
| Fig.6 | (a) topology schematic; (b) J(ξ) profiles incl. hot-runaway; (c) **D1 refined basin map**; (d) **D2 hysteresis curve** | ✅ 4 panels; standalone D1/D2 floats absorbed |

## 4. Implementation status (auto-execution)

Figures rendered under `scripts/figures_pub/`:

| ID | Figure | Cache | Status |
|---|---|---|---|
| B2 | `slow_manifold.{pdf,png}` | uses `data/fig2/cache.npz` | ✅ done |
| C1 | `homogeneous_ss.{pdf,png}` | `data/fig5/homogeneous_ss_analytic.npz`, `data/fig5/fold_curve_BiT_Schi.npz` | ✅ analytic, single-panel (PDE-overlay panel (b) retired) |
| C2 | `T_pde_vs_lsa.{pdf,png}` | `data/fig4/*` (existing) | 🗑️ RETIRED (coarse PDE grid; argument moved into §IV.B) |
| B3 | `period_scaling.{pdf,png}` | `data/fig4/fig4_grid_Bi_T_S_chi.npz` | ✅ done |
| B1 | (text + JSON) | `data/fig5/phase_timescales_WP.json` | ✅ done |
| D2 | `hysteresis_C_theta0.{pdf,png}` | `data/fig6/hysteresis_C_theta0.npz` | ✅ done |
| C3 | `SNIC_scaling.{pdf,png}` | `data/fig5/SNIC_scan_S_chi.npz` | ✅ done |
| D1 | `basin_C.{pdf,png}` | `data/fig6/basin_C_dense.npz` | ⏳ running |
| D3 | (text only — skipped figure) | n/a | ✅ done |

Section drafts (separate files for review):

| Draft | File |
|---|---|
| §IV (opening) + §IV.A | `draft_section_IV_A.tex` |
| §IV.B | `draft_section_IV_B.tex` |
| §IV.C | `draft_section_IV_C.tex` |
| §IV.D | `draft_section_IV_D.tex` |

Retirement note (2026-05-08):
- C1(b) (homogeneous_ss panel (b) — PDE oscillation overlay + C-region hatching) and C2 (T_PDE/T_LSA contrast) are dropped from §IV.C. Reason: both depend on the 25×25 (Bi_T, S_χ) PDE regime grid, which is too coarse to draw smooth contours; the physical claim ("linear stability of any homogeneous branch cannot capture a relaxation cycle on a spatially generated slow manifold") is already a direct corollary of §IV.B and does not need a noisy comparison map. §IV.C now contains C1(a) + C3 only.
- `make_homogeneous_ss.py` collapsed to single panel; `make_T_pde_vs_lsa.py` marked RETIRED in docstring (kept on disk for reproducibility); `make_preview.py` updated.

Key numerical findings (for paper integration):
- Slow-manifold S-shape: lower fold @ θ≈0.85 J≈0.19; upper fold @ θ≈3.18 J≈0.82.
- Slow:fast speed ratio in (θ,J): ≈600× across the limit cycle.
- Phase 3 (cool/quench) is the longest single phase (~40% of cycle period at WP).
- ~58% of grid cells have only the collapsed homogeneous SS as a real root; the working point and the canonical C-cell both lie inside this region.
- Bi_T(J,θ) and S_χ(J,θ) admit closed-form expressions (the SS conditions are linear in S_χ and bilinear in Bi_T after substituting u). The fold (saddle-node) locus is therefore the *image* of the single connected curve det J(F1,F2)=0 in (J,θ)-space; in (Bi_T, S_χ) it self-folds into two fold arcs meeting at a cusp — a Whitney-cusp catastrophe.
- Combined with the same closed-form map, the homogeneous-SS *root count* per (Bi_T, S_χ) pixel is obtained by counting sign-changes of Bi_T_pred(J; S_χ) - Bi_T at fixed S_χ — a 1D root-count problem per pixel. The 360×240 analytic classification (~0.4 s) shows ~88.8% of cells have a unique SS, ~10.9% have three coexisting roots (1 saddle + 2 stable), and exactly 0% have two; this odd-multiplicity-only structure follows from fold topology and corrects the original fsolve scan (which under-counted bistable cells at ~1.7% because its initial-guess set missed the hot-runaway swollen branch at θ ≈ 12, J ≈ 5).
- $T_\mathrm{PDE}/T_\mathrm{LSA}$ has median 0.45 in the LSA-Hopf-overlap region — LSA *over*-predicts the period at the collapsed-branch Hopf.
- C-cell IC threshold: $\theta_0^{\!*}\!\approx\!2.4$; below it cold-seeded ICs reach cycle (J≈2.0), above it they fall into runaway (J≈5.9).
- SNIC scaling fit RSS = 0.068 vs homoclinic RSS = 0.152 at the upper-S_χ boundary; $S_\chi^c\!\approx\!1.34$.

## 5. Intro/§III rework (deferred, parked here)

Intro currently promises:
- "characterize two distinct regimes" — actual paper finds 4 attractors;
- "contrast slab and sphere geometries" — sphere now in paper 2;
- "what parameters maximize interior reactant penetration" — also paper 2.

Will be rewritten after §IV settles.

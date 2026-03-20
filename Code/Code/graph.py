import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f, t

# ============================================================
# Parameters (edit if needed)
# ============================================================
alpha = 0.05
m = 5
n = 10

# ============================================================
# Quantiles
# ============================================================
# Identity 1: F_{alpha;m,n} = 1 / F_{1-alpha;n,m}
c_F = f.ppf(alpha, m, n)
inv_c = 1 / c_F
q_F_nm = f.ppf(1 - alpha, n, m)

# Identity 2: t_{1-alpha/2;n}^2 = F_{1-alpha;1,n}
c_t = t.ppf(1 - alpha / 2, n)
c_t_sq = c_t ** 2
q_F_1n = f.ppf(1 - alpha, 1, n)

# ============================================================
# Global style
# ============================================================
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
})

# ============================================================
# Figure 1A: X ~ F(m,n), left-tail alpha
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 5.6))

xmax1 = max(f.ppf(0.995, m, n), c_F * 3)
x1 = np.linspace(1e-4, xmax1, 2500)
y1 = f.pdf(x1, m, n)

ax.plot(x1, y1, linewidth=2.4, label=fr"$X \sim F_{{{m},{n}}}$")
mask_left = x1 <= c_F
ax.fill_between(x1[mask_left], y1[mask_left], alpha=0.30)

ax.axvline(c_F, linestyle="--", linewidth=2.2)
ax.text(c_F + 0.03, ax.get_ylim()[1] * 0.82,
        fr"$c = F_{{\alpha;{m},{n}}}$",
        rotation=90, va="top", ha="left")

ax.text(xmax1 * 0.38, ax.get_ylim()[1] * 0.22,
        fr"$P(X<c)=\alpha={alpha}$",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.set_title(r"Figure 1A. Left-tail $\alpha$ in $F_{m,n}$", pad=10)
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend(loc="upper right", frameon=True)

fig.text(0.5, 0.02, fr"$c = F_{{\alpha;{m},{n}}} = {c_F:.4f}$", ha="center")
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("fig1A_F_left_tail.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 1B: Y = 1/X ~ F(n,m), right-tail alpha
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 5.6))

xmax2 = max(f.ppf(0.995, n, m), inv_c * 1.8)
x2 = np.linspace(1e-4, xmax2, 2500)
y2 = f.pdf(x2, n, m)

ax.plot(x2, y2, linewidth=2.4, label=fr"$Y=1/X \sim F_{{{n},{m}}}$")
mask_right = x2 >= inv_c
ax.fill_between(x2[mask_right], y2[mask_right], alpha=0.30)

ax.axvline(inv_c, linestyle="--", linewidth=2.2)
ax.text(inv_c - 0.05, ax.get_ylim()[1] * 0.82, r"$1/c$",
        rotation=90, va="top", ha="right")

ax.text(xmax2 * 0.63, ax.get_ylim()[1] * 0.22,
        fr"$P(Y>1/c)=\alpha={alpha}$",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.text(xmax2 * 0.44, ax.get_ylim()[1] * 0.52,
        fr"$1/c = F_{{1-\alpha;{n},{m}}}$",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.80))

ax.set_title(r"Figure 1B. Right-tail $\alpha$ in $F_{n,m}$", pad=10)
ax.set_xlabel("y")
ax.set_ylabel("Density")
ax.legend(loc="upper right", frameon=True)

fig.text(0.5, 0.02,
         fr"$1/c = {inv_c:.4f}$ and $F_{{1-\alpha;{n},{m}}} = {q_F_nm:.4f}$",
         ha="center")
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("fig1B_F_right_tail.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 2A: T ~ t_n, central probability 1-alpha
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 5.6))

xmin3 = min(-4.5, -c_t - 1.0)
xmax3 = max(4.5, c_t + 1.0)
x3 = np.linspace(xmin3, xmax3, 2500)
y3 = t.pdf(x3, n)

ax.plot(x3, y3, linewidth=2.4, label=fr"$T \sim t_{{{n}}}$")
mask_center = (x3 >= -c_t) & (x3 <= c_t)
ax.fill_between(x3[mask_center], y3[mask_center], alpha=0.30)

ax.axvline(-c_t, linestyle="--", linewidth=2.2)
ax.axvline(c_t, linestyle="--", linewidth=2.2)

ax.text(-c_t - 0.06, ax.get_ylim()[1] * 0.80, r"$-c$",
        rotation=90, va="top", ha="right")
ax.text(c_t + 0.06, ax.get_ylim()[1] * 0.80, r"$c=t_{1-\alpha/2;n}$",
        rotation=90, va="top", ha="left")

ax.text(0, ax.get_ylim()[1] * 0.30,
        fr"$P(-c<T<c)=1-\alpha={1-alpha}$",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.set_title(r"Figure 2A. Central probability in $t_n$", pad=10)
ax.set_xlabel("t")
ax.set_ylabel("Density")
ax.legend(loc="upper right", frameon=True)

fig.text(0.5, 0.02, fr"$c=t_{{1-\alpha/2;{n}}}={c_t:.4f}$", ha="center")
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("fig2A_t_central.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Figure 2B: T^2 ~ F(1,n), left-tail 1-alpha at c^2
# ============================================================
fig, ax = plt.subplots(figsize=(8.2, 5.6))

# Start slightly away from 0 to avoid the spike dominating the whole plot
xmax4 = max(f.ppf(0.995, 1, n), c_t_sq * 1.5)
x4 = np.linspace(0.02, xmax4, 2500)
y4 = f.pdf(x4, 1, n)

ax.plot(x4, y4, linewidth=2.4, label=fr"$T^2 \sim F_{{1,{n}}}$")
mask_left2 = x4 <= c_t_sq
ax.fill_between(x4[mask_left2], y4[mask_left2], alpha=0.30)

ax.axvline(c_t_sq, linestyle="--", linewidth=2.2)
ax.text(c_t_sq + 0.06, ax.get_ylim()[1] * 0.82, r"$c^2$",
        rotation=90, va="top", ha="left")

ax.text(xmax4 * 0.56, ax.get_ylim()[1] * 0.28,
        fr"$P(T^2<c^2)=1-\alpha={1-alpha}$",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))

ax.text(xmax4 * 0.42, ax.get_ylim()[1] * 0.56,
        fr"$c^2 = F_{{1-\alpha;1,{n}}}$",
        ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.20", fc="white", ec="none", alpha=0.80))

# Optional visual cap if you want an even flatter look:
# ax.set_ylim(0, 1.2)

ax.set_title(r"Figure 2B. Left-tail $1-\alpha$ in $F_{1,n}$", pad=10)
ax.set_xlabel("x")
ax.set_ylabel("Density")
ax.legend(loc="upper right", frameon=True)

fig.text(0.5, 0.02,
         fr"$c^2=t_{{1-\alpha/2;{n}}}^2={c_t_sq:.4f}$ and $F_{{1-\alpha;1,{n}}}={q_F_1n:.4f}$",
         ha="center")
plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("fig2B_F_left_tail_from_t2.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Final console check
# ============================================================
print("Numeric checks:")
print(f"F_(alpha;{m},{n}) = {c_F:.6f}")
print(f"1 / F_(1-alpha;{n},{m}) = {1/q_F_nm:.6f}")
print(f"t_(1-alpha/2;{n})^2 = {c_t_sq:.6f}")
print(f"F_(1-alpha;1,{n}) = {q_F_1n:.6f}")
"""
Pilot Bloom Analysis — 합천창녕보
===================================
Pre-weir vs Post-weir NDCI proxy 변화 분석 + 시각화
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# ─── Load Data ───
with open(str(Path(__file__).parent.parent / "output/pilot_hapcheon_bloom.json")) as f:
    data = json.load(f)

# ─── Extract time series ───
pre = data["landsat_pre_weir"]
post = data["landsat_post_weir"]
s2 = [r for r in data["sentinel2"] if r["median"] is not None]

pre_years = [r["year"] for r in pre]
pre_medians = [r["median"] for r in pre]
post_years = [r["year"] for r in post]
post_medians = [r["median"] for r in post]
s2_years = [r["year"] for r in s2]
s2_medians = [r["median"] for r in s2]

# ─── Statistics ───
pre_mean = np.mean(pre_medians)
post_mean = np.mean(post_medians)
change = post_mean - pre_mean
pct_change = change / pre_mean * 100

print("=" * 60)
print("합천창녕보 Bloom Proxy Analysis")
print("=" * 60)
print(f"Pre-weir  (2000-2012): median NDCI proxy = {pre_mean:.4f}")
print(f"Post-weir (2013-2024): median NDCI proxy = {post_mean:.4f}")
print(f"Change: {change:+.4f} ({pct_change:+.1f}%)")
print()
print("Yearly statistics:")
print(f"  Pre-weir  range: [{min(pre_medians):.4f}, {max(pre_medians):.4f}]")
print(f"  Post-weir range: [{min(post_medians):.4f}, {max(post_medians):.4f}]")
print()

# Image counts
pre_counts = [r["count"] for r in pre]
post_counts = [r["count"] for r in post]
s2_counts = [r["count"] for r in s2]
print(f"Image counts per year:")
print(f"  Landsat pre:  {min(pre_counts)}-{max(pre_counts)} (mean {np.mean(pre_counts):.0f})")
print(f"  Landsat post: {min(post_counts)}-{max(post_counts)} (mean {np.mean(post_counts):.0f})")
print(f"  Sentinel-2:   {min(s2_counts)}-{max(s2_counts)} (mean {np.mean(s2_counts):.0f})")

# Simple t-test
from scipy import stats
t_stat, p_val = stats.ttest_ind(pre_medians, post_medians, equal_var=False)
print(f"\nWelch t-test: t={t_stat:.3f}, p={p_val:.6f}")
if p_val < 0.01:
    print("  → Highly significant (p < 0.01)")
elif p_val < 0.05:
    print("  → Significant (p < 0.05)")
else:
    print("  → Not significant")

# ─── Figure ───
fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

# Panel A: Time series
ax = axes[0]
ax.plot(pre_years, pre_medians, "o-", color="#2196F3", label="Pre-weir (Landsat)", markersize=6)
ax.plot(post_years, post_medians, "s-", color="#F44336", label="Post-weir (Landsat)", markersize=6)
ax.plot(s2_years, s2_medians, "^--", color="#4CAF50", label="Sentinel-2 NDCI", markersize=6)

# Completion year line
ax.axvline(x=2012, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(2012.2, ax.get_ylim()[1] if ax.get_ylim()[1] else 0.72, "Weir completion",
        fontsize=9, color="gray", va="top")

# Pre/post means
ax.axhline(y=pre_mean, color="#2196F3", linestyle=":", alpha=0.5)
ax.axhline(y=post_mean, color="#F44336", linestyle=":", alpha=0.5)

ax.set_xlabel("Year", fontsize=11)
ax.set_ylabel("Bloom Season Median NDCI Proxy", fontsize=11)
ax.set_title("합천창녕보 (Hapcheon-Changnyeong Weir) — Bloom Proxy Time Series",
             fontsize=13, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(1999, 2025)

# Annotation
ax.annotate(f"Pre mean: {pre_mean:.3f}",
            xy=(2006, pre_mean), xytext=(2002, pre_mean + 0.05),
            fontsize=9, color="#2196F3",
            arrowprops=dict(arrowstyle="->", color="#2196F3", alpha=0.5))
ax.annotate(f"Post mean: {post_mean:.3f}\n(+{pct_change:.1f}%)",
            xy=(2019, post_mean), xytext=(2020, post_mean + 0.05),
            fontsize=9, color="#F44336",
            arrowprops=dict(arrowstyle="->", color="#F44336", alpha=0.5))

# Panel B: Image count
ax2 = axes[1]
all_years = pre_years + post_years
all_counts = pre_counts + [r["count"] for r in post]
colors = ["#2196F3"] * len(pre_years) + ["#F44336"] * len(post_years)
ax2.bar(all_years, all_counts, color=colors, alpha=0.7, label="Landsat")
if s2_counts:
    ax2.bar(s2_years, s2_counts, color="#4CAF50", alpha=0.5, width=0.4, label="Sentinel-2")
ax2.axvline(x=2012, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Image Count", fontsize=11)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(1999, 2025)

plt.tight_layout()
out_path = str(Path(__file__).parent.parent / "output/figures/pilot_hapcheon_timeseries.png")
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\n✅ Figure saved: {out_path}")
plt.close()

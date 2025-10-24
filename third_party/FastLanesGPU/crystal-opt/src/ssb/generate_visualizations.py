#!/usr/bin/env python3

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# Find the most recent benchmark JSON file
output_dir = Path("/root/autodl-tmp/test/FLS-GPU/FastLanesGPU-main/crystal-opt/src/ssb")
json_files = sorted(glob.glob(str(output_dir / "benchmark_crystal_opt_*.json")))

if not json_files:
    print("ERROR: No benchmark JSON files found!")
    exit(1)

# Load the most recent benchmark data
json_file = Path(json_files[-1])
print(f"Loading data from: {json_file}")

with open(json_file, 'r') as f:
    data = json.load(f)

# Extract query data
queries = sorted(data['queries'].keys(), key=lambda x: (int(x[1]), int(x.split('.')[1])))
means = [data['queries'][q]['mean_ms'] for q in queries]
medians = [data['queries'][q]['median_ms'] for q in queries]
mins = [data['queries'][q]['min_ms'] for q in queries]
maxs = [data['queries'][q]['max_ms'] for q in queries]
stddevs = [data['queries'][q]['stddev_ms'] for q in queries]

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
colors = plt.cm.Set3(np.linspace(0, 1, len(queries)))

print("="*80)
print(" "*20 + "GENERATING FASTLANES GPU VISUALIZATIONS")
print("="*80)
print()

# ============================================================================
# Figure 1: Mean Execution Time Bar Chart
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 8))
bars = ax.bar(range(len(queries)), means, color=colors, alpha=0.8,
              edgecolor='black', linewidth=1.5)
ax.errorbar(range(len(queries)), means, yerr=stddevs, fmt='none', ecolor='red',
            capsize=5, capthick=2, label='Std Dev')

ax.set_xlabel('Query', fontsize=14, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('FastLanes GPU (Crystal-Opt) SSB Query Performance', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(queries)))
ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, means)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'performance_bars.png', dpi=300, bbox_inches='tight')
print("✓ Generated: performance_bars.png")

# ============================================================================
# Figure 2: Box Plot for all queries
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 8))
times_list = [data['queries'][q]['times'] for q in queries]
bp = ax.boxplot(times_list, tick_labels=queries, patch_artist=True, showmeans=True, meanline=True)

# Color the boxes
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xlabel('Query', fontsize=14, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('FastLanes GPU Query Performance Distribution', fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'performance_boxplot.png', dpi=300, bbox_inches='tight')
print("✓ Generated: performance_boxplot.png")

# ============================================================================
# Figure 3: Query Series Comparison
# ============================================================================
series_data = {
    'Q1': {'queries': [], 'means': []},
    'Q2': {'queries': [], 'means': []},
    'Q3': {'queries': [], 'means': []},
    'Q4': {'queries': [], 'means': []}
}

for q in queries:
    series = q[:2]
    series_data[series]['queries'].append(q)
    series_data[series]['means'].append(data['queries'][q]['mean_ms'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, (series, sdata) in enumerate(series_data.items()):
    ax = axes[idx]
    colors_series = plt.cm.Set2(np.linspace(0, 1, len(sdata['queries'])))
    bars = ax.bar(range(len(sdata['queries'])), sdata['means'], color=colors_series,
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Query', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (ms)', fontsize=12, fontweight='bold')
    ax.set_title(f'{series} Series Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(sdata['queries'])))
    ax.set_xticklabels(sdata['queries'], fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, sdata['means']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'series_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Generated: series_comparison.png")

# ============================================================================
# Figure 4: Coefficient of Variation
# ============================================================================
cvs = [(stddevs[i] / means[i]) * 100 for i in range(len(queries))]
fig, ax = plt.subplots(figsize=(16, 8))
bars = ax.bar(range(len(queries)), cvs, color=colors, alpha=0.8,
              edgecolor='black', linewidth=1.5)
ax.axhline(y=5.0, color='red', linestyle='--', linewidth=2, label='5% CV threshold')
ax.set_xlabel('Query', fontsize=14, fontweight='bold')
ax.set_ylabel('Coefficient of Variation (%)', fontsize=14, fontweight='bold')
ax.set_title('Query Stability Analysis - Coefficient of Variation', fontsize=16, fontweight='bold')
ax.set_xticks(range(len(queries)))
ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=11)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, cvs)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'stability_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Generated: stability_analysis.png")

# ============================================================================
# Figure 5: Summary Statistics Table
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = [['Query', 'Mean (ms)', 'Median (ms)', 'Min (ms)', 'Max (ms)', 'StdDev (ms)', 'CV (%)']]

for i, q in enumerate(queries):
    row = [
        q,
        f'{means[i]:.4f}',
        f'{medians[i]:.4f}',
        f'{mins[i]:.4f}',
        f'{maxs[i]:.4f}',
        f'{stddevs[i]:.4f}',
        f'{cvs[i]:.2f}'
    ]
    table_data.append(row)

# Add summary row
table_data.append(['', '', '', '', '', '', ''])
table_data.append(['AVERAGE',
                   f'{np.mean(means):.4f}',
                   f'{np.mean(medians):.4f}',
                   f'{np.mean(mins):.4f}',
                   f'{np.mean(maxs):.4f}',
                   f'{np.mean(stddevs):.4f}',
                   f'{np.mean(cvs):.2f}'])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.12, 0.15, 0.15, 0.15, 0.15, 0.15, 0.13])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Style header
for i in range(7):
    table[(0, i)].set_facecolor('#3498DB')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style summary rows
for i in range(7):
    table[(len(queries) + 1, i)].set_facecolor('lightgray')
    table[(len(queries) + 2, i)].set_facecolor('#FFE66D')
    table[(len(queries) + 2, i)].set_text_props(weight='bold')

# Alternate row colors
for i in range(1, len(queries) + 1):
    color = '#F0F0F0' if i % 2 == 0 else 'white'
    for j in range(7):
        table[(i, j)].set_facecolor(color)

plt.title('FastLanes GPU SSB Performance Summary Table', fontsize=16, fontweight='bold', pad=20)
plt.savefig(output_dir / 'summary_table.png', dpi=300, bbox_inches='tight')
print("✓ Generated: summary_table.png")

# ============================================================================
# Figure 6: Min-Mean-Max Range
# ============================================================================
fig, ax = plt.subplots(figsize=(16, 8))
x_pos = np.arange(len(queries))

# Plot range as error bars with mean as center
ax.errorbar(x_pos, means, yerr=[np.array(means) - np.array(mins),
                                  np.array(maxs) - np.array(means)],
            fmt='o', markersize=10, capsize=8, capthick=2,
            color='darkblue', ecolor='lightblue', elinewidth=3, alpha=0.8)

ax.set_xlabel('Query', fontsize=14, fontweight='bold')
ax.set_ylabel('Execution Time (ms)', fontsize=14, fontweight='bold')
ax.set_title('Query Performance Range (Min-Mean-Max)', fontsize=16, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir / 'performance_range.png', dpi=300, bbox_inches='tight')
print("✓ Generated: performance_range.png")

# ============================================================================
# Print Summary Statistics
# ============================================================================
print()
print("="*80)
print(" "*30 + "SUMMARY STATISTICS")
print("="*80)
print()
print(f"{'Metric':<30} {'Value':<15}")
print("-"*80)
print(f"{'Average Time (ms)':<30} {np.mean(means):<15.4f}")
print(f"{'Min Time (ms)':<30} {np.min(mins):<15.4f}")
print(f"{'Max Time (ms)':<30} {np.max(maxs):<15.4f}")
print(f"{'Total Time (ms)':<30} {np.sum(means):<15.4f}")
print(f"{'Average StdDev (ms)':<30} {np.mean(stddevs):<15.4f}")
print(f"{'Average CV (%)':<30} {np.mean(cvs):<15.2f}")
print("="*80)
print()

# Top and bottom performers
print("Top 5 Fastest Queries (by mean):")
print("-"*80)
sorted_by_speed = sorted(enumerate(means), key=lambda x: x[1])
for i, (idx, time) in enumerate(sorted_by_speed[:5]):
    print(f"  {i+1}. {queries[idx]}: {time:.4f}ms")

print()
print("Top 5 Slowest Queries (by mean):")
print("-"*80)
for i, (idx, time) in enumerate(reversed(sorted_by_speed[-5:])):
    print(f"  {i+1}. {queries[idx]}: {time:.4f}ms")

print()
print("="*80)
print()
print(f"All charts saved to: {output_dir}")
print()
print("Generated files:")
print("  1. performance_bars.png - Mean execution time with error bars")
print("  2. performance_boxplot.png - Distribution analysis")
print("  3. series_comparison.png - Performance by query series")
print("  4. stability_analysis.png - Coefficient of variation")
print("  5. summary_table.png - Complete statistics table")
print("  6. performance_range.png - Min-Mean-Max ranges")
print()
print("="*80)

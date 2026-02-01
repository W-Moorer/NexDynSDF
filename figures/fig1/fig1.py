import matplotlib.pyplot as plt
import numpy as np

# 设置字体为 Times New Roman (如果系统中没有，会回退到衬线字体)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'serif']
plt.rcParams['mathtext.fontset'] = 'stix'

# --------------------------
# 1. 数据录入 (Data Transcription)
# --------------------------
models = ['Armadillo', 'Happy', 'Frog', 'Bunny', 'Crankshaft', 'Dragon', 'Boolean', 'Temple']

# 查询时间 (Query Time) - 单位: us
query_our = [1.42, 1.98, 1.83, 0.87, 4.28, 5.1, 0.64, 0.37]
query_cgal = [31.43, 41.74, 47.18, 17.83, 68.67, 47.73, 13.76, 4.79]
query_svh = [25.29, 45.35, 40.01, 13.16, 81.5, 54.2, 30.97, 8.14]

# 构建时间 (Building Time) - 单位: s (选取Our Method的1T数据进行对比)
build_our = [151, 269, 201.12, 40.45, 605.56, 1637.76, 19.06, 21.36]
build_cgal = [0.01, 0.04, 0.01, 0.003, 0.06, 0.22, 0.001, 0.004]
build_svh = [0.87, 2.7, 0.94, 0.12, 5.3, 25.58, 0.022, 0.273]

# --------------------------
# 2. 绘图设置
# --------------------------
x = np.arange(len(models))  # 标签位置
width = 0.25  # 柱状图宽度

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- 图表 1: 查询时间 (Query Time) ---
rects1 = ax1.bar(x - width, query_our, width, label='Our Method', color='#1f77b4', alpha=0.9)
rects2 = ax1.bar(x, query_cgal, width, label='CGAL', color='#ff7f0e', alpha=0.9)
rects3 = ax1.bar(x + width, query_svh, width, label='SVH', color='#2ca02c', alpha=0.9)

ax1.set_ylabel('Query Time (us)', fontsize=14)
ax1.set_title('Comparison of Query Performance (Lower is Better)', fontsize=16, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.5)

# --- 图表 2: 构建时间 (Building Time) ---
# 注意：由于数量级差异巨大 (0.001s vs 1600s)，必须使用对数坐标
rects4 = ax2.bar(x - width, build_our, width, label='Our Method (1T)', color='#1f77b4', alpha=0.9)
rects5 = ax2.bar(x, build_cgal, width, label='CGAL', color='#ff7f0e', alpha=0.9)
rects6 = ax2.bar(x + width, build_svh, width, label='SVH', color='#2ca02c', alpha=0.9)

ax2.set_ylabel('Building Time (s) - Log Scale', fontsize=14)
ax2.set_title('Comparison of Building Time (Lower is Better)', fontsize=16, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
ax2.set_yscale('log') # 关键：设置对数坐标
ax2.legend(fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.5, which='major')

# --------------------------
# 3. 布局调整与展示
# --------------------------
plt.tight_layout()
plt.show()
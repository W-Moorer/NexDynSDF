# PyTools 工具集

本目录包含用于可视化和处理SDF（Signed Distance Field）及NSM（NexDyn Surface Mesh）文件的Python工具。

---

## 工具列表

| 工具 | 功能 | 可视化库 |
|------|------|----------|
| `nsm_reader.py` | NSM网格文件读取与可视化（含法向量） | PyVista |
| `visualize_sdf.py` | SDF零等值面可视化 | Matplotlib / Plotly |
| `visualize_sdf_pyvista.py` | SDF零等值面可视化（交互式） | PyVista |
| `visualize_sdf_pyvista_offscreen.py` | SDF零等值面可视化（离屏渲染保存图片） | PyVista |

---

## 1. nsm_reader.py - NSM网格可视化工具

### 功能
- 读取NSM（NexDyn Surface Mesh）二进制文件
- 可视化3D网格模型
- 显示顶点法向量（红色箭头）
- 支持按面片ID着色

### 依赖安装
```bash
pip install numpy pyvista
```

### 使用方法

#### 基本用法
```bash
python nsm_reader.py <nsm文件路径>
```

#### 命令行参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--no-normals` | 不显示法向量 | 显示 |
| `--normal-scale N` | 法向量箭头缩放比例 | 0.01 |
| `--normal-skip N` | 法向量显示密度（每N个三角形显示一次） | 10 |
| `--no-edges` | 不显示网格边 | 显示 |
| `--color-by-id` | 按面片ID着色 | 否 |

#### 示例
```bash
# 基本可视化
python nsm_reader.py ../models/nsm/Gear_I.nsm

# 调整法向量显示密度和大小
python nsm_reader.py ../models/nsm/Gear_I.nsm --normal-scale 0.005 --normal-skip 20

# 按面片ID着色
python nsm_reader.py ../models/nsm/Gear_I.nsm --color-by-id

# 不显示法向量，仅显示网格
python nsm_reader.py ../models/nsm/Gear_I.nsm --no-normals
```

#### 在Python代码中使用
```python
from nsm_reader import load_nsm, visualize_nsm, create_pyvista_mesh

# 加载NSM数据
data = load_nsm('../models/nsm/Gear_I.nsm')

# 访问数据
print(data.vertices)           # 顶点坐标 [N, 3]
print(data.triangles)          # 三角形索引 [M, 3]
print(data.tri_face_ids)       # 面片ID [M]
print(data.tri_vertex_normals) # 法向量 [M, 3, 3]

# 创建PyVista网格
mesh = create_pyvista_mesh(data)

# 可视化
visualize_nsm('../models/nsm/Gear_I.nsm', 
              show_normals=True, 
              normal_scale=0.01, 
              normal_skip=10)
```

### 交互操作
- **左键拖动**: 旋转视角
- **右键拖动**: 缩放
- **中键拖动**: 平移
- **滚轮**: 缩放
- **'q'**: 退出

---

## 2. visualize_sdf.py - SDF零等值面可视化（Matplotlib/Plotly）

### 功能
- 从SdfSampler生成的.raw文件加载SDF数据
- 使用Marching Cubes算法提取零等值面
- 支持Matplotlib静态可视化和Plotly交互式可视化
- 支持两个SDF对比显示

### 依赖安装
```bash
pip install numpy scipy matplotlib plotly scikit-image
```

### 使用方法

#### 命令行参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `files` | SDF原始数据文件路径（一个或多个） | 必需 |
| `-l, --level` | 等值面值 | 0.0 |
| `-t, --titles` | 每个SDF的标题 | 自动命名 |
| `-s, --static` | 使用Matplotlib静态可视化 | Plotly交互式 |
| `-c, --compare` | 对比模式（并排显示两个SDF） | 否 |

#### 示例
```bash
# 可视化单个文件（Plotly交互式）
python visualize_sdf.py approx_sdf.raw

# 可视化单个文件（Matplotlib静态）
python visualize_sdf.py approx_sdf.raw -s

# 对比两个SDF
python visualize_sdf.py approx_sdf.raw exact_sdf.raw -c

# 使用不同的等值面值
python visualize_sdf.py approx_sdf.raw -l 0.1

# 自定义标题
python visualize_sdf.py approx_sdf.raw exact_sdf.raw -t "Approximate" "Exact"
```

### 工作流程
1. 使用SdfSampler工具采样SDF数据：
   ```bash
   SdfSampler.exe Gear_ApproxSDF_High.bin approx_sdf.raw 128
   SdfSampler.exe Gear_ExactSDF_High.bin exact_sdf.raw 128
   ```
2. 运行可视化脚本：
   ```bash
   python visualize_sdf.py approx_sdf.raw exact_sdf.raw -c
   ```

---

## 3. visualize_sdf_pyvista.py - SDF零等值面可视化（PyVista交互式）

### 功能
- 使用PyVista进行高性能3D可视化
- 支持多个SDF并排对比
- 支持体数据切片可视化
- 网格平滑处理

### 依赖安装
```bash
pip install numpy pyvista
```

### 使用方法

#### 命令行参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `files` | SDF原始数据文件路径（一个或多个） | 必需 |
| `-l, --level` | 等值面值 | 0.0 |
| `-t, --titles` | 每个SDF的标题 | 自动命名 |
| `--no-smooth` | 禁用网格平滑 | 启用平滑 |
| `--volume` | 使用体数据切片模式 | 等值面模式 |
| `--axis` | 切片轴（x/y/z） | z |

#### 示例
```bash
# 可视化单个文件
python visualize_sdf_pyvista.py gear_sampled.raw

# 对比两个文件
python visualize_sdf_pyvista.py approx_sdf.raw exact_sdf.raw

# 使用不同的等值面值
python visualize_sdf_pyvista.py gear_sampled.raw -l 0.1

# 禁用网格平滑
python visualize_sdf_pyvista.py gear_sampled.raw --no-smooth

# 体数据切片可视化
python visualize_sdf_pyvista.py gear_sampled.raw --volume --axis z
```

### 交互操作
- **左键拖拽**: 旋转
- **右键拖拽**: 缩放
- **中键拖拽**: 平移
- **滚轮**: 缩放
- **'q' 或关闭窗口**: 退出

---

## 4. visualize_sdf_pyvista_offscreen.py - SDF可视化（离屏渲染）

### 功能
- 不打开交互窗口，直接生成静态图片
- 适用于批量处理或服务器环境
- 支持自定义等值面值和标题

### 依赖安装
```bash
pip install numpy pyvista
```

### 使用方法

#### 命令行参数
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `file` | SDF原始数据文件路径 | 必需 |
| `-o, --output` | 输出图片路径 | 必需 |
| `-l, --level` | 等值面值 | 0.0 |
| `-t, --title` | 标题 | "SDF" |

#### 示例
```bash
# 基本用法
python visualize_sdf_pyvista_offscreen.py gear_sampled.raw -o gear.png

# 指定等值面值和标题
python visualize_sdf_pyvista_offscreen.py sphere_sampled.raw -o sphere.png -l 0.0 -t "Sphere SDF"
```

---

## 文件格式说明

### NSM文件格式
NSM（NexDyn Surface Mesh）是自定义的二进制网格文件格式：

| 字段 | 大小 | 说明 |
|------|------|------|
| magic | 4 bytes | "NSM\0" |
| version | 4 bytes | 版本号（当前=1） |
| num_vertices | 4 bytes | 顶点数量 |
| num_triangles | 4 bytes | 三角形数量 |
| reserved | 48 bytes | 保留字段 |
| vertices | num_vertices × 3 × 8 bytes | 顶点坐标（double） |
| triangles | num_triangles × 3 × 4 bytes | 三角形索引（uint32） |
| tri_face_ids | num_triangles × 4 bytes | 面片ID（uint32） |
| tri_vertex_normals | num_triangles × 3 × 3 × 8 bytes | 顶点法向量（double） |

### SDF RAW文件格式
由SdfSampler工具生成的原始SDF数据文件：

| 字段 | 大小 | 说明 |
|------|------|------|
| grid_resolution | 4 bytes | 网格分辨率（int） |
| bbox_min | 12 bytes | 边界框最小值（3×float） |
| bbox_max | 12 bytes | 边界框最大值（3×float） |
| grid_data | N³ × 4 bytes | 体数据（float32） |

---

## 常见问题

### Q: 安装PyVista时遇到问题？
A: PyVista依赖VTK，可能需要额外安装：
```bash
pip install vtk pyvista
```

### Q: 可视化窗口无法打开？
A: 确保您的系统支持OpenGL，或尝试使用离屏渲染版本。

### Q: 如何调整法向量显示密度？
A: 使用 `--normal-skip` 参数，值越大显示的箭头越少：
```bash
python nsm_reader.py model.nsm --normal-skip 50
```

### Q: 如何保存可视化结果为图片？
A: 使用离屏渲染工具：
```bash
python visualize_sdf_pyvista_offscreen.py data.raw -o output.png
```

---

## 更新日志

- **2025-02-02**: 添加 `nsm_reader.py` - NSM网格可视化工具
- **2025-02-01**: 添加 `visualize_sdf_pyvista_offscreen.py` - 离屏渲染版本
- **2025-01-31**: 添加 `visualize_sdf_pyvista.py` - PyVista交互式可视化
- **2025-01-30**: 添加 `visualize_sdf.py` - 基础SDF可视化工具

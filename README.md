# NexDynSDF

> 本项目基于 [SdfLib](https://github.com/UPC-ViRVIG/SdfLib) 进行开发，感谢 UPC-ViRVIG 团队提供的优秀开源实现。

NexDynSDF 是一个高性能的有符号距离场（Signed Distance Field, SDF）计算库，支持从三角网格（OBJ/VTP/NSM格式）生成自适应八叉树SDF、精确八叉树SDF和混合八叉树SDF。

## 功能特性

### 自适应八叉树SDF (OctreeSdf)
基于论文 *"Adaptive approximation of signed distance fields through piecewise continuous interpolation"* (Computers & Graphics 114, 2023) 的完整实现：

- **插值方法**：
  - 三线性插值 (Trilinear)：8个系数，保证 $C^0$ 连续
  - 三三次插值 (Tricubic)：64个系数，支持 $C^1$ 连续（默认）

- **误差估计策略**：
  - 梯形法则 (Trapezoidal Rule)：基于 $\{0, 0.5, 1\}^3$ 采样点的数值积分
  - 辛普森法则 (Simpson's Rule)：更高精度的数值积分
  - 距离衰减规则 (By Distance)：基于到表面距离的自适应误差阈值

- **构建算法**：
  - `CONTINUITY`：BFS构建，强制跨层级连续性（推荐）
  - `NO_CONTINUITY`：DFS构建，无连续性保证但更快
  - `UNIFORM`：均匀细分到最大深度

- **连续性保证**：
  - 同尺寸邻居：天然 $C^0$/$C^1$ 连续
  - 跨层级邻居：通过"匹配邻居插值"强制连续性

### 精确八叉树SDF (ExactOctreeSdf)
基于论文 *"Triangle Influence Supersets for Fast Distance Computation"* (CGF 2023) 的实现：

- **三角形影响区域**：使用GJK算法计算三角形影响超集
- **快速剔除**：基于包围盒和GJK距离计算的三角形筛选
- **精确查询**：使用 TriangleMeshDistance 库进行精确距离计算

### 混合八叉树SDF (HybridOctreeSdf)
**核心思想**：构建阶段使用 Nagata Patch 计算更平滑的顶点值，查询阶段保持八叉树的快速检索。

- **优势**：
  - 构建时保留尖锐特征，同时提升曲面光滑性。
  - 查询速度接近标准 OctreeSdf。
- **数据依赖**：
  - 需要 `EnhancedNagataData`（由工具自动计算并缓存为 `.eng` 文件）。
  - 推荐输入 `.nsm` 以利用面法向信息。

### 支持的输入格式
- **OBJ**：Wavefront OBJ 三角网格文件
- **VTP**：VTK XML PolyData 格式文件
- **NSM**：NexDyn Surface Mesh 二进制格式（支持每面片法线信息）

### 输出格式
- 二进制序列化格式（使用 Cereal 库），支持快速加载

### 性能优化
- OpenMP 多线程并行计算
- AVX2/SSE 向量化指令集优化
- 依赖缓存机制，避免重复下载

## 项目结构

```
NexDynSDF/
├── CMakeLists.txt              # CMake 构建配置
├── vcpkg.json                  # vcpkg 依赖清单
├── LICENSE                     # 许可证文件
├── .gitignore                  # Git 忽略配置
├── include/
│   └── sdflib/
│       ├── SdfFunction.h               # SDF 基类
│       ├── OctreeSdf.h                 # 自适应八叉树SDF
│       ├── ExactOctreeSdf.h            # 精确八叉树SDF
│       ├── ExactOctreeSdfDepthFirst.h  # 精确八叉树DFS构建
│       ├── HybridOctreeSdf.h           # 混合八叉树SDF
│       ├── NagataTrianglesInfluenceForBuild.h # 混合构建用影响区域策略
│       ├── TriangleMeshDistance.h      # ICG 精确距离查询库
│       ├── TrianglesInfluence.h        # 三角形影响区域计算
│       ├── InterpolationMethods.h      # 插值方法（三线性/三三次）
│       ├── OctreeSdfUtils.h            # 误差估计函数
│       ├── OctreeSdfBreadthFirst.h     # BFS构建+连续性
│       ├── OctreeSdfBreadthFirstNoDelay.h # BFS构建（无延迟版本）
│       ├── OctreeSdfDepthFirst.h       # DFS构建
│       └── utils/
│           ├── Mesh.h                  # 网格加载工具（OBJ/VTP）
│           ├── MeshBinaryLoader.h      # NSM 二进制格式加载工具
│           ├── NagataEnhanced.h        # Nagata 裂隙边增强及几何查询
│           ├── NagataPatch.h           # 基础 Nagata Patch 数据结构
│           ├── InterpolationMethods.h  # 插值方法工具
│           ├── TriangleUtils.h         # 三角形工具
│           ├── GJK.h                   # GJK算法头文件
│           ├── GJK.inl                 # GJK算法内联实现
│           ├── BoundingBox.h           # 包围盒工具
│           ├── Timer.h                 # 计时器工具
│           └── UsefullSerializations.h # 序列化辅助工具
├── src/
│   ├── SdfFunction.cpp         # SDF基类实现
│   ├── OctreeSdf.cpp           # 八叉树SDF实现
│   ├── OctreeSdfUniform.cpp    # 均匀八叉树实现
│   ├── ExactOctreeSdf.cpp      # 精确八叉树SDF实现
│   ├── main.cpp                # 主程序入口（示例）
│   ├── OctreeSdfBreadthFirst.h      # BFS构建内部实现
│   ├── OctreeSdfBreadthFirstNoDelay.h # BFS无延迟构建内部实现
│   ├── OctreeSdfDepthFirst.h   # DFS构建内部实现
│   └── utils/
│       ├── Mesh.cpp            # 网格加载实现
│       ├── TriangleUtils.cpp   # 三角形工具实现
│       ├── GJK.cpp             # GJK算法实现
│       └── Timer.cpp           # 计时器实现
├── tools/
│   ├── SdfExporter/
│   │   └── main.cpp            # SDF导出工具主程序
│   └── SdfSampler/
│       └── main.cpp            # SDF空间采样工具主程序
├── pytools/
│   ├── README.md                       # Python工具文档
│   ├── nsm_reader.py                   # NSM网格读取与可视化
│   ├── nagata_patch.py                 # Nagata曲面插值计算模块
│   ├── nagata_storage.py               # Nagata数据存储工具
│   ├── check_nagata_cracks.py          # Nagata裂缝检测工具
│   ├── visualize_nagata.py             # Nagata曲面可视化
│   ├── visualize_obj_nagata.py         # OBJ文件Nagata可视化
│   ├── visualize_sdf.py                # Matplotlib/Plotly可视化脚本
│   ├── visualize_sdf_pyvista.py        # PyVista交互式可视化脚本
│   └── visualize_sdf_pyvista_offscreen.py  # PyVista离屏渲染脚本
├── tests/
│   ├── test_all.cpp            # 综合单元测试
│   ├── test_hybrid_accuracy.cpp # 混合SDF精度测试
│   ├── test_openmp.cpp         # OpenMP测试
│   ├── test_spdlog.cpp         # spdlog日志测试
│   ├── test_icg.cpp            # ICG距离查询测试
│   ├── test_fcpw.cpp           # FCPW库测试
│   ├── test_enoki.cpp          # Enoki向量化测试
│   ├── test_eigen.cpp          # Eigen矩阵库测试
│   ├── test_cereal.cpp         # Cereal序列化测试
│   └── test_glm.cpp            # GLM数学库测试
├── demos/
│   ├── extract_junction_nsm.py      # 连接处提取演示
│   ├── extract_two_faces_nsm.py     # 双面提取演示
│   └── generate_cone_nsm.py         # 圆锥生成演示
├── models/
│   ├── nsm/                    # NSM格式模型
│   │   ├── Gear_I.nsm
│   │   ├── Gear_I.eng          # 对应的增强数据
│   │   ├── Gear_II.nsm
│   │   └── Gear_II.eng
│   └── vtp/                    # VTP格式模型
│       ├── complex_geometry/   # 复杂几何体
│       ├── nonsmooth_geometry/ # 非光滑几何体
│       ├── smooth_geometry/    # 光滑几何体
│       └── smooth_geometry_check/ # 光滑几何体验证
└── third_party/                # FetchContent 缓存目录
    ├── enoki_lib-src/          # Enoki向量化库
    └── fcpw_lib-src/           # FCPW最近点查询库
```

## 依赖项

### 系统依赖

- CMake >= 3.20
- C++17 兼容编译器
- OpenMP
- vcpkg (建议安装在 `C:/vcpkg`)
- Git

### 库依赖

#### C++ 依赖

| 库名 | 版本 | 获取方式 | 用途 |
|------|------|----------|------|
| glm | >= 0.9.9.8 | vcpkg | 向量数学库 |
| spdlog | >= 1.12.0 | vcpkg | 日志输出 |
| cereal | >= 1.3.2 | vcpkg | 二进制序列化 |
| eigen3 | >= 3.4.0 | vcpkg | 矩阵运算 |
| enoki | 2a18afa | FetchContent | 向量化优化 |
| fcpw | dd65ec2 | FetchContent | 快速最近点查询 |

#### Python 可视化依赖（可选）

```powershell
# PyVista 可视化（推荐）
pip install numpy pyvista

# Matplotlib/Plotly 可视化（备选）
pip install numpy scipy matplotlib plotly scikit-image
```

## 构建说明

### 环境配置

确保已设置 `VCPKG_ROOT` 环境变量：

```powershell
# PowerShell 临时设置
$env:VCPKG_ROOT = "C:\vcpkg"

# 或者永久设置（系统环境变量）
[Environment]::SetEnvironmentVariable("VCPKG_ROOT", "C:\vcpkg", "User")
```

### 首次构建

```powershell
# 创建构建目录
mkdir build
cd build

# 配置项目（使用 vcpkg 工具链）
cmake .. -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"

# 构建
cmake --build . --config Release
```

或者使用 Visual Studio：

```powershell
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
cmake --build . --config Release
```

### 依赖缓存机制

本项目使用两种缓存策略避免重复下载依赖：

1. **vcpkg Manifest 模式**: glm、spdlog、cereal、eigen3 通过 `vcpkg.json` 管理，安装后缓存在 `vcpkg_installed` 目录

2. **FetchContent 缓存**: enoki 和 fcpw 缓存在 `third_party/` 目录，通过 `FETCHCONTENT_BASE_DIR` 配置

删除 `build/` 目录后重新配置时，依赖不会重新下载，配置时间从约 100 秒缩短到约 5 秒。

### 清理重建

```powershell
# 删除构建目录（依赖缓存保留）
Remove-Item -Recurse -Force build

# 重新构建
mkdir build
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
cmake --build . --config Release
```

### 运行

```powershell
# 运行 SDF 导出工具
.\Release\SdfExporter.exe

# 运行 SDF 采样工具
.\Release\SdfSampler.exe input.bin output.raw 128

# 运行测试
.\Release\test_all.exe
```

## vcpkg Manifest 模式说明

本项目使用 `vcpkg.json` 文件声明依赖，这是 vcpkg 的 **manifest 模式**。在此模式下：

1. **自动安装依赖**：CMake 配置时会自动下载并安装 `vcpkg.json` 中声明的所有库
2. **版本控制**：通过 `builtin-baseline` 锁定依赖版本，确保构建可复现
3. **项目隔离**：依赖安装在项目目录下的 `vcpkg_installed` 文件夹中

### vcpkg.json 结构

```json
{
  "name": "nexdynsdf",
  "version": "0.1.0",
  "dependencies": [
    "glm",
    "spdlog",
    "cereal",
    "eigen3"
  ]
}
```

## 工具说明

### SdfExporter 工具

将三角网格（OBJ/VTP/NSM）转换为SDF二进制文件。

**位置**: `build/Release/SdfExporter.exe`

**用法**:

```
SdfExporter <input> <output.bin> [options]

参数:
  <input>               输入网格文件 (.obj, .vtp, .nsm)
  <output.bin>          输出SDF二进制文件

Options:
  --depth <n>              八叉树深度 (默认: 8)
  --start_depth <n>        起始深度 (默认: 1)
  --algorithm <type>       算法: continuity, no_continuity, uniform (默认: continuity)
  --sdf_format <format>    SDF格式: octree, exact_octree, hybrid (默认: octree)
  --termination <threshold> 终止阈值 (默认: 1e-3)
  --num_threads <n>        线程数 (默认: 1)
  --help                   显示帮助信息
```

> **注意**: 使用 `hybrid` 格式时：
> 1. 仅支持 `.nsm` 输入。
> 2. 程序会自动计算或加载 `.eng` (Enhanced Data) 缓存文件。

### SdfSampler 工具

将SDF二进制文件采样为均匀网格的RAW格式，便于可视化和后续处理。

**位置**: `build/Release/SdfSampler.exe`

**用法**:

```
SdfSampler <input_sdf_file> <output_raw_file> [grid_resolution]

参数:
  input_sdf_file    : SDF二进制文件路径
  output_raw_file   : 输出RAW文件路径
  grid_resolution   : 网格分辨率（默认: 128）
```

**输出文件格式**:

| 字段 | 类型 | 大小 | 说明 |
|------|------|------|------|
| gridRes | int32 | 4 bytes | 网格分辨率（立方体） |
| bbox_min.x | float32 | 4 bytes | 包围盒最小X |
| bbox_min.y | float32 | 4 bytes | 包围盒最小Y |
| bbox_min.z | float32 | 4 bytes | 包围盒最小Z |
| bbox_max.x | float32 | 4 bytes | 包围盒最大X |
| bbox_max.y | float32 | 4 bytes | 包围盒最大Y |
| bbox_max.z | float32 | 4 bytes | 包围盒最大Z |
| grid_data | float32[] | gridRes³ × 4 bytes | 距离场数据 |

### NagataExporter 工具

基于NSM文件导出增强Nagata细分结果为OBJ。

**位置**: `build/Release/NagataExporter.exe`

**用法**:

```
NagataExporter <input.nsm> <output_dir> <subdivision_level> [tolerance]

参数:
  <input.nsm>           NSM输入文件
  <output_dir>          OBJ输出目录
  <subdivision_level>   细分级别 (1,2,3...)
  [tolerance]           k_factor (默认 0.1)
```

**输出命名**:
`{inputStem}_enhanced_L{subdivisionLevel}.obj`

**Python读取示例**:

```python
import struct
import numpy as np

def load_raw_sdf(filepath):
    with open(filepath, 'rb') as f:
        grid_res = struct.unpack('i', f.read(4))[0]
        bbox_min = struct.unpack('fff', f.read(12))
        bbox_max = struct.unpack('fff', f.read(12))
        num_voxels = grid_res ** 3
        grid_data = np.frombuffer(f.read(num_voxels * 4), dtype=np.float32)
        grid_data = grid_data.reshape((grid_res, grid_res, grid_res))
    return grid_data, bbox_min, bbox_max
```

### 使用示例

#### 1. 生成SDF文件

```powershell
# 生成自适应八叉树SDF（默认使用三三次插值+连续性）
.\SdfExporter models/obj/bunny.obj output/bunny_octree.bin --depth 8 --algorithm continuity

# 生成精确八叉树SDF
.\SdfExporter models/obj/bunny.obj output/bunny_exact.bin --sdf_format exact_octree --depth 8

# 生成 混合 SDF（推荐用于需要平滑但保留特征的模型）
.\SdfExporter models/nsm/Gear_I.nsm output/gear_hybrid.bin --sdf_format hybrid --depth 8

# 使用多线程加速
.\SdfExporter models/obj/bunny.obj output/bunny.bin --depth 8 --num_threads 8
```

#### 2. 空间采样（用于可视化）

```powershell
# 基本用法：将SDF文件采样为均匀网格
.\SdfSampler output/bunny_octree.bin output/bunny_sampled.raw 128

# 参数说明：
#   参数1: 输入SDF文件路径
#   参数2: 输出RAW文件路径
#   参数3: 网格分辨率（默认128，可选）
```

采样后的 `.raw` 文件格式：
- **Header** (28 bytes): gridRes(int) + bbox_min(3×float) + bbox_max(3×float)
- **Data** (gridRes³ × float): 距离场数据

#### 3. 可视化

使用 PyVista 进行3D可视化：

```powershell
# 交互式可视化
python pytools/visualize_sdf_pyvista.py output/bunny_sampled.raw

# 生成静态图片（离屏渲染）
python pytools/visualize_sdf_pyvista_offscreen.py output/bunny_sampled.raw -o output/bunny.png

# 对比两个SDF
python pytools/visualize_sdf_pyvista.py output/bunny_approx.raw output/bunny_exact.raw
```

使用 Matplotlib/Plotly 可视化（备选）：

```powershell
# 静态可视化
python pytools/visualize_sdf.py output/bunny_sampled.raw --static

# 交互式可视化
python pytools/visualize_sdf.py output/bunny_sampled.raw

# 对比两个SDF
python pytools/visualize_sdf.py approx.raw exact.raw --compare
```

#### 4. NSM文件可视化

```powershell
# 可视化NSM网格及其法向量
python pytools/nsm_reader.py models/nsm/Gear_I.nsm

# 按面片ID着色
python pytools/nsm_reader.py models/nsm/Gear_I.nsm --color-by-id

# 调整法向量显示
python pytools/nsm_reader.py models/nsm/Gear_I.nsm --normal-scale 0.005 --normal-skip 20
```

#### 5. Nagata增强验证

```powershell
# C++导出
.\NagataExporter models/nsm/Gear_I.nsm output/nagata_cpp 2 0.1

# Python导出
python pytools/nagata_exporter.py models/nsm/Gear_I.nsm output/nagata_py --levels 1 2 3 --tolerance 0.1

# C++/Python对比验证
python pytools/validate_nagata.py --nsm models/nsm/Gear_I.nsm --cpp .\NagataExporter --output output/nagata_validation --levels 1 2 3 --tolerance 0.1

# 叠加可视化
python pytools/visualize_obj_nagata.py output/nagata_cpp/Gear_I_enhanced_L2.obj --overlay output/nagata_py/Gear_I_enhanced_L2.obj
```

#### 6. 完整工作流程示例

```powershell
# 步骤1: 从OBJ生成SDF
.\SdfExporter models/obj/Gear.obj output/gear.bin --depth 6 --algorithm continuity

# 步骤2: 空间采样（256x256x256分辨率）
.\SdfSampler output/gear.bin output/gear_sampled.raw 256

# 步骤3: 可视化
python pytools/visualize_sdf_pyvista_offscreen.py output/gear_sampled.raw -o output/gear.png
```

### 在代码中使用

```cpp
#include <sdflib/SdfFunction.h>
#include <sdflib/OctreeSdf.h>
#include <sdflib/utils/Mesh.h>

// 加载网格
sdflib::Mesh mesh("path/to/model.vtp");

// 获取包围盒并添加边距
sdflib::BoundingBox box = mesh.getBoundingBox();
box.addMargin(0.1f * box.getSize().x);

// 创建八叉树SDF（使用三三次插值和连续性）
sdflib::OctreeSdf sdf(mesh, box, 
    /*depth=*/8, 
    /*startDepth=*/1,
    /*terminationThreshold=*/1e-3f,
    /*algorithm=*/sdflib::OctreeSdf::InitAlgorithm::CONTINUITY,
    /*numThreads=*/4);

// 查询距离
glm::vec3 point(0.0f, 0.0f, 0.0f);
float distance = sdf.getDistance(point);

// 保存到文件
sdf.saveToFile("output.sdf");

// 从文件加载
auto loadedSdf = sdflib::SdfFunction::loadFromFile("output.sdf");
```

## 算法说明

### 自适应八叉树SDF (OctreeSdf)

基于论文 *"Adaptive approximation of signed distance fields through piecewise continuous interpolation"* (Computers & Graphics 114, 2023) 的完整实现：

#### 1. 插值多项式

**三线性插值 (Trilinear)**：
- 8个系数，对应单位立方体8个顶点的值
- 保证 $C^0$ 连续性
- 公式：$g(x,y,z) = \sum_{i=0}^{1}\sum_{j=0}^{1}\sum_{k=0}^{1} a_{ijk}\,x^{i}y^{j}z^{k}$

**三三次插值 (Tricubic)**：
- 64个系数，基于 Lekien & Marsden 的约束体系
- 每个顶点8个值：值 + 梯度(3) + 二阶导(3) + 三阶导(1)
- 支持 $C^1$ 连续性（跨层级）
- 公式：$g(x,y,z) = \sum_{i=0}^{3}\sum_{j=0}^{3}\sum_{k=0}^{3} a_{ijk}\,x^{i}y^{j}z^{k}$

#### 2. 误差估计

使用数值求积近似 RMSE：

$$
\mathrm{RMSE}(g) = \sqrt{\int_0^1\int_0^1\int_0^1 (g-f)^2 \,dx\,dy\,dz}
$$

**梯形法则**：
- 采样点：$\{0, 0.5, 1\}^3$，共27个点
- 额外查询：19次（排除8个顶点，因为顶点处 $g=f$）
- 权重：张量积形式 $(1,2,1) \otimes (1,2,1) \otimes (1,2,1)$

**辛普森法则**：
- 更高精度的数值积分
- 使用相同的采样点但不同权重

#### 3. 连续性强制

**问题**：不同深度邻居共享面/边时，非共享顶点会导致不连续

**解决方案**：
1. 使用BFS（广度优先）按层构建
2. 第一遍：计算误差，决定哪些节点细分
3. 第二遍：
   - 不细分的节点：写入为叶子
   - 细分的节点：创建子节点，对共享面/边上的非共享顶点，使用邻居的插值结果（而非真实场值）

**梯度缩放**：
- 世界坐标梯度需乘以节点尺寸 $L$
- 二阶导数乘以 $L^2$
- 三阶导数乘以 $L^3$

### 精确八叉树SDF (ExactOctreeSdf)

基于论文 *"Triangle Influence Supersets for Fast Distance Computation"* (CGF 2023) 的实现：

#### 1. 三角形影响区域

使用GJK算法计算每个三角形的影响超集：
- 对节点8个顶点计算到三角形的距离范围
- 通过GJK距离计算确定最小/最大距离
- 筛选出可能影响节点内任意点的三角形

#### 2. 构建流程

1. 计算所有三角形的预计算数据（法线、包围盒等）
2. 自顶向下遍历八叉树
3. 对每个节点：
   - 使用GJK计算三角形影响超集
   - 筛选出可能包含最近三角形的候选集
   - 对8个顶点计算精确距离
4. 达到最大深度时存储为叶子

#### 3. 查询优化

- 使用 TriangleMeshDistance 库（ICG算法）
- 支持精确的有符号距离计算
- 角度加权伪法线用于符号判定

### 混合八叉树SDF (HybridOctreeSdf)

构建阶段利用 Nagata Patch 计算更平滑的顶点值，查询阶段保留八叉树的快速检索：

- 输入为 `.nsm` 时可直接利用面法向信息
- 通过 `.eng` 文件缓存裂隙边增强数据
- 构建与查询兼顾质量与性能

## 实现细节

### 插值方法切换

默认使用三三次插值，可在 `OctreeSdf.cpp` 中切换：

```cpp
// 三线性插值（8系数，C0连续）
typedef TriLinearInterpolation InterpolationMethod;

// 三三次插值（64系数，C1连续）- 默认
typedef TriCubicInterpolation InterpolationMethod;
```

### 终止规则

```cpp
// 梯形法则 - 默认
OctreeSdf::TerminationRule::TRAPEZOIDAL_RULE

// 辛普森法则
OctreeSdf::TerminationRule::SIMPSONS_RULE

// 距离衰减规则
OctreeSdf::TerminationRule::BY_DISTANCE_RULE

// 无限制（细分到最大深度）
OctreeSdf::TerminationRule::NONE
```

### 多线程支持

使用 `initOctreeWithContinuityNoDelay` 算法支持多线程：

```cpp
OctreeSdf sdf(mesh, box, depth, startDepth, 
    OctreeSdf::TerminationRule::TRAPEZOIDAL_RULE,
    TerminationRuleParams::setTrapezoidalRuleParams(1e-3f),
    OctreeSdf::InitAlgorithm::CONTINUITY,
    /*numThreads=*/8);  // 使用8线程
```

## 测试

```powershell
cd build
ctest --output-on-failure
```

或直接运行测试程序：

```powershell
# 综合测试
.\Release\test_all.exe

# 混合SDF测试
.\Release\test_hybrid_accuracy.exe

# 依赖库测试
.\Release\test_glm.exe
.\Release\test_eigen.exe
.\Release\test_cereal.exe
.\Release\test_spdlog.exe
.\Release\test_enoki.exe
.\Release\test_fcpw.exe
.\Release\test_icg.exe
.\Release\test_openmp.exe
```

## 常见问题

### 1. vcpkg 依赖下载慢

可以设置国内镜像或代理：

```powershell
# 设置代理
$env:HTTP_PROXY = "http://127.0.0.1:7890"
$env:HTTPS_PROXY = "http://127.0.0.1:7890"
```

### 2. 找不到 vcpkg 工具链

确保 `VCPKG_ROOT` 环境变量正确设置，并在 CMake 配置时指定工具链文件：

```powershell
cmake .. -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT\scripts\buildsystems\vcpkg.cmake"
```

### 3. FetchContent 缓存不生效

确保 `FETCHCONTENT_BASE_DIR` 在 `include(FetchContent)` 之前设置。查看 `CMakeLists.txt` 中的配置顺序。

### 4. 三三次插值内存占用大

三三次插值每个叶子节点存储64个float（256字节），比三线性的8个float（32字节）大8倍。对于内存敏感的应用，可切换为三线性插值。

### 5. Hybrid SDF 构建失败

- 确保输入文件存在且格式正确
- 检查是否有写权限生成 `.eng` 缓存文件
- Hybrid 仅支持 NSM 输入

## 参考文献

1. **Adaptive Octree SDF**: 
   - *"Adaptive approximation of signed distance fields through piecewise continuous interpolation"*, Computers & Graphics 114, 2023

2. **Exact Octree SDF**:
   - *"Triangle Influence Supersets for Fast Distance Computation"*, Computer Graphics Forum (CGF) 2023

3. **Tricubic Interpolation**:
   - Lekien & Marsden, *"Tricubic interpolation in three dimensions"*

4. **TriangleMeshDistance**:
   - José Antonio Fernández Fernández, *"Triangle Mesh Distance"* (MIT License)

5. **Nagata Patch**:
   - Nagata, *"Local Interpolation for Curve and Surface Construction"*, 2005

## 许可证

本项目基于参考项目迁移开发，遵循原始代码的许可证条款。

## 致谢

本项目基于 [SdfLib](https://github.com/UPC-ViRVIG/SdfLib) 项目进行开发，感谢 UPC-ViRVIG 团队提供的优秀开源实现。NexDynSDF 在 SdfLib 的基础上进行了改进和扩展，包括：
- 混合SDF构建支持
- NSM二进制格式支持
- 裂隙边处理
- 更多可视化工具

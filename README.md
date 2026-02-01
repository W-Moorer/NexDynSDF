# NexDynSDF

> 本项目基于 [SdfLib](https://github.com/UPC-ViRVIG/SdfLib) 进行开发，感谢 UPC-ViRVIG 团队提供的优秀开源实现。

NexDynSDF 是一个高性能的有符号距离场（Signed Distance Field, SDF）计算库，支持从三角网格（OBJ/VTP格式）生成自适应八叉树SDF和精确八叉树SDF。

## 功能特性

- **自适应八叉树SDF (OctreeSdf)**: 基于自适应细分的八叉树结构，支持多种初始化算法
  - `CONTINUITY`: 保证SDF连续性的算法
  - `NO_CONTINUITY`: 不保证连续性的快速算法
  - `UNIFORM`: 均匀细分算法

- **精确八叉树SDF (ExactOctreeSdf)**: 基于 ICG (Improved Closest Point Query) 算法的精确距离查询

- **支持的输入格式**:
  - OBJ 三角网格文件
  - VTP (VTK PolyData) 文件

- **输出格式**: 二进制序列化格式（使用 Cereal 库），支持快速加载

- **性能优化**:
  - OpenMP 多线程并行计算
  - AVX2/SSE 向量化指令集优化
  - 依赖缓存机制，避免重复下载

## 项目结构

```
NexDynSDF/
├── CMakeLists.txt          # CMake 构建配置
├── vcpkg.json              # vcpkg 依赖清单
├── include/
│   └── sdflib/
│       ├── SdfFunction.h          # SDF 基类
│       ├── OctreeSdf.h            # 自适应八叉树SDF
│       ├── ExactOctreeSdf.h       # 精确八叉树SDF
│       ├── TriangleMeshDistance.h # ICG 距离查询库
│       └── utils/
│           ├── Mesh.h             # 网格加载工具
│           ├── BoundingBox.h      # 包围盒工具
│           └── Timer.h            # 计时器工具
├── src/
│   ├── SdfFunction.cpp
│   ├── OctreeSdf.cpp
│   ├── ExactOctreeSdf.cpp
│   └── utils/
│       ├── Mesh.cpp
│       └── Timer.cpp
├── tools/
│   └── SdfExporter/
│       └── main.cpp        # SDF导出工具主程序
├── tests/
│   └── test_all.cpp        # 单元测试
└── third_party/            # FetchContent 缓存目录
```

## 依赖项

### 系统依赖

- CMake >= 3.20
- C++17 兼容编译器
- OpenMP
- vcpkg (安装在 `C:/vcpkg`)
- Git

### 库依赖

| 库名 | 版本 | 获取方式 | 用途 |
|------|------|----------|------|
| glm | >= 0.9.9.8 | vcpkg | 向量数学库 |
| spdlog | >= 1.12.0 | vcpkg | 日志输出 |
| cereal | >= 1.3.2 | vcpkg | 二进制序列化 |
| eigen3 | >= 3.4.0 | vcpkg | 矩阵运算 |
| enoki | 2a18afa | FetchContent | 向量化优化 |
| fcpw | dd65ec2 | FetchContent | 快速最近点查询 |

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

## 使用说明

### SdfExporter 工具

```
SdfExporter <input.vtp> <output.bin> [options]

Options:
  --depth <n>              八叉树深度 (默认: 8)
  --start_depth <n>        起始深度 (默认: 1)
  --algorithm <type>       算法: continuity, no_continuity, uniform (默认: continuity)
  --sdf_format <format>    SDF格式: octree, exact_octree (默认: octree)
  --termination <threshold> 终止阈值 (默认: 1e-3)
  --num_threads <n>        线程数 (默认: 1)
  --help                   显示帮助信息
```

### 使用示例

```powershell
# 生成自适应八叉树SDF
.\SdfExporter models/obj/bunny.obj output/bunny_octree.bin --depth 8 --algorithm continuity

# 生成精确八叉树SDF
.\SdfExporter models/obj/bunny.obj output/bunny_exact.bin --sdf_format exact_octree --depth 8

# 使用多线程加速
.\SdfExporter models/obj/bunny.obj output/bunny.bin --depth 8 --num_threads 8
```

### 在代码中使用

```cpp
#include <sdflib/SdfFunction.h>
#include <sdflib/OctreeSdf.h>
#include <sdflib/utils/Mesh.h>

// 加载网格
sdflib::Mesh mesh("path/to/model.obj");

// 获取包围盒并添加边距
sdflib::BoundingBox box = mesh.getBoundingBox();
box.addMargin(0.1f * box.getSize().x);

// 创建八叉树SDF
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
sdflib::OctreeSdf loadedSdf;
loadedSdf.loadFromFile("output.sdf");
```

## 算法说明

### 自适应八叉树SDF

基于论文 "Adaptive Sparse Octree SDF" 的实现，通过自适应细分平衡精度和存储：

- 在表面附近使用较深的八叉树层级
- 在远离表面区域使用较浅的层级
- 支持多种细分策略以适应不同应用场景

### 精确八叉树SDF

基于 ICG (Improved Closest Point Query) 算法，使用 `TriangleMeshDistance.h` 实现：

- 精确计算点到三角网格的有符号距离
- 支持内部/外部判断
- 适合需要高精度距离查询的应用

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

## 测试

```powershell
cd build
ctest --output-on-configuration
```

或直接运行测试程序：

```powershell
.\Release\test_all.exe
```

## 许可证

本项目基于参考项目迁移开发，遵循原始代码的许可证条款。

## 致谢

本项目基于 [SdfLib](https://github.com/UPC-ViRVIG/SdfLib) 项目进行开发，感谢 UPC-ViRVIG 团队提供的优秀开源实现。NexDynSDF 在 SdfLib 的基础上进行了以下改进和扩展：

- 使用 vcpkg manifest 模式管理依赖，优化依赖安装流程
- 引入 FetchContent 缓存机制，避免重复下载依赖
- 扩展输入格式支持（OBJ/VTP）
- 优化构建配置和工具链支持

### 使用的第三方库

- [Enoki](https://github.com/mitsuba-renderer/enoki) - 向量化库
- [FCPW](https://github.com/rohan-sawhney/fcpw) - 快速最近点查询库
- [ICG](https://github.com/InteractiveComputerGraphics) - 距离查询算法

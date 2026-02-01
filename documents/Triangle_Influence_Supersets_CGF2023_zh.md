# Triangle Influence Supersets for Fast Distance Computation（CGF 2023）中文整理（去图保公式）

> 论文：*Triangle Influence Supersets for Fast Distance Computation*，Eduard Pujol，Antonio Chica，**Computer Graphics Forum 42(6), 2023**。  
> 本文档：按原文章节结构做中文整理，**去除图片与表格**，保留关键定义、公式与推导思路，并补充一份“完整计算路径（build/query pipeline）”。

---

## 摘要（Abstract）

本文提出一种用于三角网格（triangle mesh）**精确**查询有符号距离场（Signed Distance Field, SDF）的加速结构。方法基于空间离散（octree）：对每个空间节点，仅存储在该区域内会影响最近距离计算的三角形集合，从而显著降低“最近三角形搜索”的代价，并避免对距离场的近似。核心在于：对每个节点，给定一个三角形 $T_c$，构造一个**保守的影响区域超集**，用于判定其他三角形在该节点内是否“必然冗余”。影响区域不需要显式构造；相交测试通过凸集支持函数与迭代优化实现。作者证明方法正确性，并与其它精确方法对比，显示查询速度更快。

---

# 完整计算路径（实现视角的 Pipeline）

本节不是原文独立章节，而是将全文方法串起来，便于实现。

## A. 预处理 / 构建（Build）

**输入**：封闭、可定向的流形三角网格（假设可用于确定 inside/outside）。  
**输出**：octree；每个叶节点带一个三角形列表（或压缩编码），保证对叶内任意点 $p$：
$$
\arg\min_T d(p,T)\ \in\ \text{LeafTriangles}(p)
$$

构建流程（自顶向下）：

1. **初始化根节点** $R_0$（AABB 覆盖网格），将全部三角形集合赋给根节点列表。
2. 对当前节点 $R$：
   - 若达到最大深度或三角形数 $\le$ 阈值：设为叶子并停止。
   - 否则：将节点划分为 8 个子节点（octree subdivision），每个子节点先继承父节点三角形列表，然后进行**三角形丢弃（discard）**以缩小列表。
3. **丢弃规则（核心）**：对节点 $R$，选少量“代表性三角形”作为候选遮挡者（通常来自 8 个角点的最近三角形）。若对某个三角形 $T_f$ 能找到候选 $T_c$ 使得
$$
T_f \cap R^{CH}_{T_c} = \varnothing,
$$
   则 $T_f$ 在 $R$ 内必不可能成为最近三角形，可丢弃（保守正确：只会少丢，不会错丢）。
4. 将过滤后的列表写入子节点并递归。

（可选）**内存压缩**：对深层节点用 bit-vector 表示“相对父列表的保留子集”，减少重复存储。

## B. 查询（Query）

给定查询点 $p$：

1. 在 octree 中找到包含 $p$ 的叶节点 $R(p)$（或落到最深可用节点）。
2. 枚举叶节点三角形列表 $\mathcal{T}=\text{LeafTriangles}(p)$，计算
$$
d(p)=\min_{T\in\mathcal{T}} d(p,T),
$$
   并记录最近三角形/最近特征用于符号。
3. 计算符号（inside/outside）：可用最近特征的伪法线/一致法线等常见方法（依赖网格封闭可定向的假设）。
4. 输出 SDF：$\text{SDF}(p)=\pm d(p)$。

---

# 1. Introduction（引言）

## 1.1 问题背景

对三角网格的 SDF 查询，本质是两步：
1) 求最近距离：距离等于到**最近三角形**的距离；  
2) 决定符号：对封闭可定向表面，可由最近点局部法线/伪法线确定 inside/outside。

传统精确方法常使用层次包围结构（如 BVH）做最近三角形搜索。但当查询点远离表面时，搜索范围扩大，遍历代价上升。

## 1.2 本文目标

通过**空间离散 + 节点局部三角形列表**，将每次查询的候选三角形数量压到很小，从而优先优化查询性能，并保持精确（不近似距离）。

---

# 2. Previous Work（相关工作）

相关路线可粗分为：

1) **预离散/近似距离场**：如 uniform grid / narrow band、OpenVDB、Fast Marching 等。通常构建/查询很快，但存在误差或需要额外策略确保精度与一致性。

2) **精确最近距离加速**：如 BVH、球层次结构、空间分割结构等，保持精确但查询代价受“查询点与表面距离”影响较大。

本文方法属于“精确加速”，但通过为每个空间节点构造“影响三角形超集”，将查询候选严格限制在一个局部集合内。

---

# 3. Outline（方法概览与关键定义）

本章给出核心几何对象：节点影响区域、影响超集、以及如何用它们丢弃三角形。

## 3.1 空间节点与目标性质

令 $R$ 是 octree 的一个节点（轴对齐盒子）。我们希望在 $R$ 内的所有点 $x$，其最近三角形一定属于节点存储集合 $\mathcal{T}_R$。

当从父节点下放到子节点时，初始可令子节点继承父节点集合，然后在子节点内**丢弃不可能成为最近的三角形**。

## 3.2 “理想影响区域” $R^{*}_{T_c}$

对某个候选三角形 $T_c$，定义集合 $S(x,r)$ 为以 $x$ 为球心、半径 $r$ 的闭球。令 $d(x,T_c)$ 是点到三角形的无符号距离。定义：

$$
R^{*}_{T_c}=\bigcup_{x\in R} S\bigl(x,\ d(x,T_c)\bigr).
$$

直观上：对每个 $x\in R$，把“以 $x$ 为中心、半径等于 $x$ 到 $T_c$ 的距离”的球取并。

**关键用途**：若另一三角形 $T_f$ 不与 $R^{*}_{T_c}$ 相交，则对 $R$ 内任意点 $x$，$T_f$ 都不可能比 $T_c$ 更近，因此 $T_f$ 可在节点 $R$ 中视为冗余并丢弃。

但直接构造/测试 $R^{*}_{T_c}$ 代价高。

## 3.3 影响区域超集 $R^{CH}_{T_c}$（Triangle Influence Superset）

令节点 $R$ 的 8 个角点为 $c_{ijk}$，其中 $i,j,k\in\{0,1\}$。定义每个角点到三角形 $T_c$ 的距离
$$
d_{ijk}=d(c_{ijk},T_c).
$$
构造 8 个球 $S(c_{ijk}, d_{ijk})$，定义其并的凸包：

$$
R^{CH}_{T_c}=\operatorname{CH}\left(\ \bigcup_{i,j,k\in\{0,1\}} S\bigl(c_{ijk},\ d(c_{ijk},T_c)\bigr)\ \right).
$$

其中 $\operatorname{CH}(\cdot)$ 表示凸包。

**核心性质（保守性）**：
$$
R^{CH}_{T_c}\supseteq R^{*}_{T_c}.
$$
证明见附录 Appendix A。由此可得：若
$$
T_f \cap R^{CH}_{T_c} = \varnothing,
$$
则一定也有 $T_f \cap R^{*}_{T_c}=\varnothing$，从而 $T_f$ 在节点 $R$ 内冗余可丢弃。  
这保证“不会错丢”，最多只会“少丢”。

---

# 4. Intersection Test（相交/距离阈值测试）

本章解决：如何高效判断 $T_f$ 与 $R^{CH}_{T_c}$ 是否相交（或距离是否小于阈值），且**无需显式构造** $R^{CH}_{T_c}$。

## 4.1 用 CSO（Minkowski Difference）把相交转化为“原点包含”

对两个凸集 $A,B$，其配置空间障碍（CSO）/ Minkowski 差为
$$
A\ominus B=\{a-b\mid a\in A,\ b\in B\}.
$$
经典结论：
$$
A\cap B\neq\varnothing\quad\Longleftrightarrow\quad \mathbf{0}\in (A\ominus B).
$$

令 $A=R^{CH}_{T_c}$，$B=T_f$。判断相交等价于判断原点是否在 $A\ominus B$ 内。

## 4.2 支持函数（Support Mapping）

对凸集 $C$ 定义支持点
$$
s_C(\mathbf{v})=\arg\max_{\mathbf{x}\in C} \mathbf{v}\cdot \mathbf{x}.
$$
则 CSO 的支持函数可写为：
$$
s_{A\ominus B}(\mathbf{v})=s_A(\mathbf{v})-s_B(-\mathbf{v}).
$$

### 4.2.1 三角形的支持点

对三角形 $T_f$，支持点就是 3 个顶点中使 $\mathbf{v}\cdot \mathbf{x}$ 最大的顶点。

### 4.2.2 $R^{CH}_{T_c}$ 的支持点（无需构造凸包）

$R^{CH}_{T_c}$ 是“8 个球并的凸包”。凸包的支持点可由组成集合的极值给出，因此可计算：

- 单个球 $S(\mathbf{c},r)$ 的支持点为
$$
s_{sphere}(\mathbf{v})=\mathbf{c}+r\frac{\mathbf{v}}{\|\mathbf{v}\|}.
$$
- 对 8 个角点球分别求支持点，取使点积最大的那个作为 $s_{R^{CH}_{T_c}}(\mathbf{v})$。

于是 CSO 支持点：
$$
s_{CSO}(\mathbf{v}) = s_{R^{CH}_{T_c}}(\mathbf{v}) - s_{T_f}(-\mathbf{v}).
$$

## 4.3 用 Frank–Wolfe 近似“原点到 CSO 的距离”

判断 $\mathbf{0}\in D$（$D$ 为 CSO）等价于求 $\min_{\mathbf{x}\in D}\|\mathbf{x}\|$ 是否为 0。  
令目标函数：
$$
f(\mathbf{x})=\|\mathbf{x}\|,\qquad \nabla f(\mathbf{x})=\frac{\mathbf{x}}{\|\mathbf{x}\|}.
$$
Frank–Wolfe 迭代：
$$
\mathbf{x}_{n+1}=\mathbf{x}_n+\alpha\left(s_D(-\nabla f(\mathbf{x}_n))-\mathbf{x}_n\right).
$$

其中下降方向
$$
\mathbf{d}=s_D(-\nabla f(\mathbf{x}_n))-\mathbf{x}_n,
$$
作者给出最优步长（把原点投影到直线段上并截断到 $[0,1]$）：

$$
\alpha=\min\left(1,\ \frac{-\mathbf{x}_n\cdot \mathbf{d}}{\|\mathbf{d}\|^2}\right).
$$

## 4.4 两类停止条件（加速）

在实现中，我们通常并非要求“精确判定包含”，而是用于“是否冗余可丢弃”的保守测试，因此可加入阈值停止：

### 4.4.1 半径阈值（in-radius）停止

设阈值 $\delta>0$，当
$$
\|\mathbf{x}_n\|<\delta
$$
即可停止并判定“在半径内”（可认为相交/距离足够小）。阈值的设置可与后续的“侵蚀（erosion）”结合。

### 4.4.2 分离轴（out-of-radius）停止

记当前迭代使用的方向为 $\mathbf{v}$（通常为负梯度方向）。若在方向 $\mathbf{v}$ 上满足：支持点在该方向上与原点的距离已经超过 $\delta$，并且“支持点在该方向上先于原点出现”，则可判定 $D$ 不可能进入以原点为中心半径 $\delta$ 的球，提前停止（本质是一个分离轴证据）。

## 4.5 侵蚀（erosion）技巧：用 $d_{\min}$ 作为阈值

令
$$
d_{\min}=\min_{i,j,k} d(c_{ijk},T_c).
$$
将每个角点球半径减去 $d_{\min}$（等价于对 $R^{CH}_{T_c}$ 做球形侵蚀），得到 $\tilde{R}^{CH}_{T_c}$。此时测试可转化为：

- 判断 $T_f$ 到 $\tilde{R}^{CH}_{T_c}$ 的距离是否小于 $d_{\min}$；
- 在 Frank–Wolfe 中直接使用阈值 $\delta=d_{\min}$，更容易快速提前停止。

实现上无需显式构造侵蚀形状：只需在支持点计算中把半径改为
$$
\tilde d_{ijk}=d_{ijk}-d_{\min}.
$$

---

# 5. Discarding Triangles（三角形丢弃策略）

单节点内若对所有三角形两两比较会导致 $O(m^2)$ 代价（$m$ 为节点三角形数）。本章给出启发式以降低比较次数。

## 5.1 只用 8 个“角点最近三角形”作为候选 $T_c$

对节点 $R$ 的每个角点 $c_{ijk}$，在节点三角形列表中找到最近三角形 $T_{ijk}$。最多得到 8 个候选集合
$$
\{T_{ijk}\}_{i,j,k\in\{0,1\}}.
$$
丢弃时仅用这些候选作遮挡者（$T_c$），去测试其余三角形是否冗余。

这会造成“少丢”（列表偏大），但所有被丢弃的都仍然是保守正确的。

## 5.2 将待测三角形 $T_f$ 分配到最近角点（1-corner centroid heuristic）

对于每个待测三角形 $T_f$：

1) 计算 $T_f$ 的质心 $g(T_f)$；  
2) 找到离质心最近的角点 $c_{ijk}$；  
3) 仅用该角点对应的候选三角形 $T_{ijk}$ 做一次相交测试：  
   - 若 $T_f\cap R^{CH}_{T_{ijk}}=\varnothing$，丢弃 $T_f$；  
   - 否则保留。

该启发式在论文实验中表现最好（比较次数少且丢弃效果好）。

## 5.3（可实现的）节点处理伪代码（高层）

- 计算每个角点的最近三角形 $T_{ijk}$；  
- 对每个三角形 $T_f$：
  - 选最近角点 $c_{ijk}$；
  - 用 Intersection Test 判断 $T_f$ 是否与 $R^{CH}_{T_{ijk}}$ 相交；
  - 不相交则丢弃。

---

# 6. Results（结果与观察）

（本节去除原文图表与具体数值，仅保留主要结论与实现要点。）

1) **查询速度**：构建较慢，但单次查询通常达到微秒级；与常见 BVH/SVH 等精确方法相比，尤其在“查询点远离表面”时优势显著。  
2) **构建成本**：方法偏向“以构建换查询”；适用于需要大量 SDF 查询的场景（采样、偏移面、碰撞检测迭代等）。  
3) **参数影响**：叶节点最大三角形数、最大深度、Frank–Wolfe 最大迭代次数会共同影响构建时间、内存与查询速度。  
4) **压缩存储**：可用层级合并 + bit-vector 表示深层节点相对父节点的三角形子集，从而显著降低内存开销，同时保持查询开销可控。

---

# 7. Conclusions（结论）

本文提出了一个用于精确距离场查询的 octree 加速结构：通过“影响区域超集”保守地筛除每个节点内不可能成为最近的三角形，使得查询阶段只需对少量三角形做点-三角形距离计算。影响区域无需显式构造，而是通过支持函数与迭代优化完成相交/阈值判定。该方法在大量查询的应用中可获得显著加速。

---

# Appendix A：Influence Supersets（影响超集的推导要点）

本附录目标：证明
$$
R^{CH}_{T_c}\supseteq R^{*}_{T_c}.
$$
作者通过构造中间集合 $R^{+}_{T_c}$ 分两步完成：
$$
R^{+}_{T_c}\supseteq R^{*}_{T_c},\qquad R^{CH}_{T_c}\supseteq R^{+}_{T_c}.
$$

## A.1 定义三线性插值与 $R^{+}_{T_c}$

设 $TriInt(\alpha,\beta,\gamma,\mathbf{v}_{ijk})$ 表示对八个角点值 $\mathbf{v}_{ijk}$ 的三线性插值（$\alpha,\beta,\gamma\in[0,1]$）。这里 $\mathbf{v}_{ijk}$ 既可以是标量（例如 $d_{ijk}$），也可以是向量（例如角点位置 $c_{ijk}$）。

定义：
- 角点位置：$c_{ijk}$；
- 角点距离：$d_{ijk}=d(c_{ijk},T_c)$。

构造集合：

$$
R^{+}_{T_c}
=\bigcup_{\alpha,\beta,\gamma\in[0,1]}
S\!\Bigl(
TriInt(\alpha,\beta,\gamma,c_{ijk}),\ 
TriInt(\alpha,\beta,\gamma,d_{ijk})
\Bigr).
$$

解释：对盒子内部任一点（由 $c_{ijk}$ 插值得到），用角点距离的插值作为球半径，取并得到 $R^{+}_{T_c}$。

## A.2 证明 $R^{+}_{T_c}\supseteq R^{*}_{T_c}$ 的关键不等式

关键需要证明插值半径不小于真实距离，即对任意 $\alpha,\beta,\gamma\in[0,1]$：

$$
TriInt(\alpha,\beta,\gamma,d_{ijk})\ \ge\ d\!\bigl(TriInt(\alpha,\beta,\gamma,c_{ijk}),\ T_c\bigr).
\tag{A1}
$$

证明思路（按原文推导链条概括）：

1) 对每个角点 $c_{ijk}$，取其到三角形 $T_c$ 的最近点
$$
q_{ijk}=\arg\min_{\mathbf{x}\in T_c}\|\mathbf{x}-c_{ijk}\|.
$$
于是
$$
d_{ijk}=\|q_{ijk}-c_{ijk}\|.
$$

2) 利用“向量三线性插值的范数不超过范数的三线性插值”这一性质（可视为 Minkowski/凸性相关不等式在三线性组合下的形式），得到：

$$
\bigl\|TriInt(\alpha,\beta,\gamma,q_{ijk})-TriInt(\alpha,\beta,\gamma,c_{ijk})\bigr\|
\ \le\
TriInt(\alpha,\beta,\gamma,\|q_{ijk}-c_{ijk}\|).
$$

3) 右侧 $TriInt(\alpha,\beta,\gamma,\|q_{ijk}-c_{ijk}\|)=TriInt(\alpha,\beta,\gamma,d_{ijk})$；左侧可写为两点距离：

$$
d\!\bigl(TriInt(\alpha,\beta,\gamma,q_{ijk}),\ TriInt(\alpha,\beta,\gamma,c_{ijk})\bigr).
$$

4) 注意 $TriInt(\alpha,\beta,\gamma,q_{ijk})$ 在三角形上（因为 $q_{ijk}\in T_c$ 且三线性组合保持在其仿射包络相关集合中；原文据此建立与 $T_c$ 的距离关系），因此：

$$
d\!\bigl(TriInt(\alpha,\beta,\gamma,q_{ijk}),\ TriInt(\alpha,\beta,\gamma,c_{ijk})\bigr)
\ \ge\
d\!\bigl(TriInt(\alpha,\beta,\gamma,c_{ijk}),\ T_c\bigr).
$$

将以上不等式链条合并即可得到 (A1)，从而 $R^{+}_{T_c}\supseteq R^{*}_{T_c}$ 成立。

## A.3 证明 $R^{CH}_{T_c}\supseteq R^{+}_{T_c}$（凸包覆盖）

取任意点 $p\in R^{+}_{T_c}$，则存在 $\alpha,\beta,\gamma$ 与向量 $\mathbf{v}$ 使得：

$$
p=TriInt(\alpha,\beta,\gamma,c_{ijk})+\mathbf{v},
\qquad
\|\mathbf{v}\|\le TriInt(\alpha,\beta,\gamma,d_{ijk}).
$$

构造 8 个角点球中的向量 $\mathbf{v}_{ijk}$，使其方向与 $\mathbf{v}$ 相同且 $\|\mathbf{v}_{ijk}\|=d_{ijk}$，从而：

- 每个点 $c_{ijk}+\mathbf{v}_{ijk}\in S(c_{ijk},d_{ijk})$；
- 且通过三线性插值可得到
$$
TriInt(\alpha,\beta,\gamma,c_{ijk}+\mathbf{v}_{ijk})
=TriInt(\alpha,\beta,\gamma,c_{ijk})+TriInt(\alpha,\beta,\gamma,\mathbf{v}_{ijk}).
$$
并使 $TriInt(\alpha,\beta,\gamma,\mathbf{v}_{ijk})=\mathbf{v}$（满足长度界与方向约束）。

因此 $p$ 可以表示为角点球内点的凸组合（或等价地落在其凸包内），从而 $p\in R^{CH}_{T_c}$，即 $R^{CH}_{T_c}\supseteq R^{+}_{T_c}$。

综上：
$$
R^{CH}_{T_c}\supseteq R^{+}_{T_c}\supseteq R^{*}_{T_c}.
$$

---

## 参考实现提示（可选）

- 点到三角形距离 $d(p,T)$：标准 closest point on triangle（面内、边、顶点三种情况）。
- 支持点：
  - 三角形：3 顶点取 $\max \mathbf{v}\cdot \mathbf{x}$；
  - 球：$\mathbf{c}+r\frac{\mathbf{v}}{\|\mathbf{v}\|}$；
  - $R^{CH}$：8 个球支持点取最大。
- Frank–Wolfe：迭代上限可固定（如 10–20）；若未收敛，为保守性可“按相交处理”（即不丢弃）。
- 侵蚀：用 $d_{\min}$ 减半径，并将阈值设为 $\delta=d_{\min}$。


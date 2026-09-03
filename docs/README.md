# Path-tracing Monte Carlo Library for 3D Radiative Transfer

> 论文：*Path-tracing Monte Carlo Library for 3D Radiative Transfer in Highly Resolved Cloudy Atmospheres*  
> 作者：Najda Villefranque, Richard Fournier, Fleur Couvreux 等  
> 来源：JAMES（arXiv:1902.01137v2, 2019）  
> 本地文件：`Path-tracing Monte Carlo Library for 3D Radiative.pdf`

---

## 1. 问题

### 1.1 背景

云与辐射的相互作用是数值天气预报、气候模拟以及遥感反演的核心难点之一。云在全球辐射平衡中起关键作用（整体冷却效应），出现频率高，且在时空上高度复杂。

### 1.2 核心矛盾


| 维度              | 现状                      | 问题                                     |
| --------------- | ----------------------- | -------------------------------------- |
| **3D 辐射传输**     | 能准确描述云-辐射三维相互作用         | 计算成本高，难以业务化                            |
| **1D 辐射传输**     | 业务系统中广泛使用               | 在破碎云场等条件下是**明显错误近似**（云侧缘效应被忽略）         |
| **高分辨率 LES 数据** | 现代大涡模拟可达 5 m 分辨率、数十亿网格点 | 传统 MC 路径追踪对体数据分辨率**高度敏感**，后处理与辐射分析难以扩展 |




### 1.3 路径追踪的三重瓶颈

云辐射精确路径追踪长期面临**精度、速度、规模**三者难以兼得的「不可能三角」。以下三个瓶颈相互耦合，构成恶性循环（见 1.3.4）。

#### 1.3.1 Beer 定律的非线性 → 必须逐体素穿越

参与性介质中的辐射传输遵循 RTE。沿路径 s 的辐亮度可写为：

```
L(s) = L(0) · exp(−τ(s)) + 源项积分,   τ = ∫ σₑ ds
```

光学厚度 τ 出现在**指数衰减项**内，这是体渲染与表面渲染的根本分野：

| 线性场景（表面） | 非线性场景（体） |
|------------------|------------------|
| 路径贡献可预计算、缓存 | 每步须重算 exp(−τ) |
| 求交是离散的（击中/未击中） | 贡献是沿路径的连续积分 |
| BVH 可跳过空白区域 | 跳过体素即丢失 τ 贡献 |
| 路径可独立采样后加权 | 权重与全路径累积 τ 耦合 |

直观地说：表面渲染是「0/1 开关」，体渲染是「整条路径上的积分」——在未知路径全长之前，无法提前算出 exp(−τ)。

**计算代价估算**（LES 1000³ 网格）：路径平均穿越约 L ~ 1000 个体素，每步需三线性插值、消光查表与指数运算。传统 MC 每像素 10⁴ ~ 10⁶ 条路径，总操作量可达 10¹³ ~ 10¹⁵ 量级；分辨率每翻一倍，穿越次数近似翻倍。

#### 1.3.2 表面加速 vs 体数据加速 — Embree 无法直接迁移

Embree 代表的表面加速范式：

```
构建 BVH → 光线-三角形求交 → O(log N)
```

核心是**空间索引 + 稀疏性**：大部分空间为空，光线可「跳跃」穿过。

体数据为何套不上同一套逻辑？

| 表面 BVH | 体数据需求 |
|----------|------------|
| 稀疏（几何仅占表面） | 密集（每体素均有光学属性） |
| 离散求交（有/无） | 连续贡献（每体素贡献 Δτ） |
| 可跳过空白 | 不能随意跳过 |
| 复杂度 O(log N) | 本质 O(L)，L 为路径长度 |

即便建立体素八叉树，**云核心**（高光学厚度）仍需密集采样；**云边界**（低光学厚度）虽可粗化，却恰是辐射梯度最剧烈处——自适应采样在此往往两头不讨好。

| 变通方案 | 思路 | 主要局限 |
|----------|------|----------|
| Delta-tracking | 用最大消光做虚拟散射，跳过部分体素 | 效率依赖 max/min 消光比；云中可达 10³，无效步过多 |
| Beer 映射 | 沿路径预积分 τ 存为 3D 纹理 | 视角变化需重算，难实时 |
| 光子映射 | 预计算光场 | 动态 LES 时序无法复用 |

#### 1.3.3 数据规模 — 38 GB 超单机内存，必须 Out-of-Core

论文 Meso-NH 算例的数据量：

| 参数 | 值 |
|------|-----|
| 空间网格 | 1000³ ≈ 10⁹ 体素 |
| 光谱 quadrature（可见光 k-distribution） | ×30 |
| 原始场（NetCDF） | **≈ 38 GB** |
| 八叉树加速结构（VTK） | **≈ 7.4 GB** |

加上温度、压力、水汽等辅助场，总量轻松突破 50 GB，远超典型单机内存。

Out-of-core 的核心困难：

| 挑战 | 说明 |
|------|------|
| I/O 瓶颈 | 路径追踪是**随机访问**（光线方向任意），难以顺序预读，Cache 命中率低 |
| 分页颠簸 | 随机换页导致单条路径可能触发数百次磁盘 I/O |
| GPU 显存 | 多卡总量有限，且 HBM 带宽仍低于片上访问需求 |
| 分布式分片 | 光线跨节点分片时通信开销大 |

| Out-of-core 思路 | 局限 |
|------------------|------|
| 分块光栅化 | 块边界误差；视角依赖 |
| 流式条带加载 | 多视角时缓存爆炸 |
| 压缩表示 | 解压开销 + 精度损失 |
| Morton/Z-order 排序 | 对随机方向光线局部性提升有限 |

#### 1.3.4 三者的恶性循环

```
LES 分辨率 ↑
    → 体素数 ↑ → 逐体素穿越代价 ↑        （瓶颈 1）
    → 内存需求 ↑ → 必须 Out-of-Core      （瓶颈 3）
    → 随机 I/O 加剧                      （瓶颈 3）
    → 无法沿用 BVH 式「跳跃」加速         （瓶颈 2）
    → 回到逐体素穿越                     （瓶颈 1）
```

这正是云辐射模拟长期依赖二流近似、δ-Eddington 等简化方案，而非精确路径追踪的结构性原因。**本文库的核心贡献**即针对此三角：零碰撞算法打破瓶颈 1 与 2 的绑定，八叉树 + Morton 块存储缓解瓶颈 3（详见第 2、3 节）。

### 1.4 经典简化方案：二流近似与 δ-Eddington

在精确路径追踪不可行时，业务与气候模式普遍采用**解析近似**求解 RTE。二者是同一思路谱系上的两级：二流近似最粗，δ-Eddington 针对云滴前向散射做了关键修正。

#### 1.4.1 为什么要简化？

平面平行大气中的 RTE（角度形式）可写为：

```
μ · dI/dτ = I − (ω/2) ∫₋₁¹ I(μ') P(μ, μ') dμ'
```

其中 I 为辐射强度，ω 为单次散射反照率，P 为散射相函数。**计算难点在 P**：云滴 Mie 散射极度前向集中（尖锐前向峰），若严格离散角度积分，代价与 MC 路径追踪同属重量级。二流近似与 δ-Eddington 是两种在**角度维度上降维**的经典技巧。

#### 1.4.2 二流近似（Two-Stream Approximation）

**物理直觉：** 将全天空各方向的光简并为两束——向下（I⁻，μ < 0）与向上（I⁺，μ > 0）。

**数学操作：** 假设辐射强度仅依赖天顶角符号，将角度积分方程约化为 I⁺、I⁻ 两个常微分方程组成的方程组，可进一步导出层间递推的**解析公式**。

| 优点 | 局限 |
|------|------|
| 极快，可嵌入 GCM 每一时间步 | 丢失全部角度细节 |
| 有闭式解，便于参数化开发 | 不区分直射与漫射 |
| ecRad TripleClouds 等方案的基础 | 只能估算通量总量，无法描述多次散射路径结构 |

#### 1.4.3 δ-Eddington 近似（Delta-Eddington Approximation）

**物理直觉：** 云滴前向散射峰极窄——大量光子几乎「直直往前冲」而未真正偏转。δ-Eddington 的核心洞见：**将这部分前向散射从散射过程中剥离，视作未散射的直接传输**。

**数学操作：** 将相函数拆分为 δ 峰 + 平滑剩余部分：

```
P(cos Θ) = 2f · δ(1 − cos Θ) + (1 − f) · P_Eddington(cos Θ)
```

- **2f · δ(1 − cos Θ)**：前向尖峰，归入直接光通道
- **(1 − f) · P_Eddington**：剩余部分用 Eddington 线性角分布近似处理

**等效变换（重正化）：** 拆分后物理参数被修正——

- 光学厚度 τ **减小**（前向传输不再计入消光阻碍）
- 单次散射反照率 ω **降低**（前向散射份额被抽离）

此即论文与 ecRad 对比实验中常用的 **delta-Eddington 缩放**（Joseph et al., 1976）：对 Henyey-Greenstein 相函数做同样处理，以修正二流求解器对前向峰的过度反射。

| 优点 | 局限 |
|------|------|
| 对前向散射介质精度远高于纯二流 | 仍是一阶/二阶矩近似 |
| 计算仍极快（解析层间递推） | 强各向异性场景（大太阳天顶角、侧向散射）偏差明显 |
| 云辐射参数化的事实标准之一 | direct/diffuse 划分本身可能带偏（见 §5.5） |

#### 1.4.4 与本文路径追踪库的定位关系

```
精确 MC 路径追踪（本文库）  ←——  无角度降维，可采样真实 Mie/HG 相函数
         ↑ 验证基准
δ-Eddington + 二流（ecRad 等）  ←——  业务可用，但 1D/3D 参数化仍有系统偏差
         ↑ 仍太慢 / 太大
三重瓶颈（§1.3）  ←——  迫使社区长期停留于简化方案
```

论文 §5.5 实验即沿此定位展开：以 MC + 真实 Mie 相函数为参考，检验 ecRad（TripleClouds / SPARTACUS）及 delta-Eddington 缩放 HG 的 direct/total 通量比偏差。

---



## 2. 研究机理



### 2.1 零碰撞算法（Null-collision / Maximum Cross Section）

在真实消光系数 k 稀疏的区域添加**虚拟碰撞体**，构造变换场：

```
k̂ = k + kₙ
```

使 k̂ 在全域或分块区域内近似均匀。此时：

- Beer 定律可在均匀 k̂ 场上直接采样自由程（指数分布）
- 采样到碰撞后，以概率 k/k̂ 判定为**真实碰撞**，否则为**零碰撞**（前向散射等价于无散射）
- 算法**无偏**（不引入近似），但路径递归深度增加

**关键理论意义**：零碰撞将 Beer 指数内的积分移出，使体数据也能使用与表面相同的层次加速结构。

### 2.2 层次八叉树（Octree）加速网格

在变换场 k̂ 上构建**自适应八叉树**：

- 递归合并 2³ 体素组，合并判据基于**光学厚度** τ̃
- 每个合并体素存储局部 k_min、k_max，用于零碰撞拒绝采样
- 目标三要素：
  1. 局部调整 k̂，最小化零碰撞拒绝次数
  2. 加速 k̂ 场穿越，碰撞时快速访问真实 k 值
  3. 预计算成本可控



### 2.3 积分形式 ↔ 算法 双向映射

库设计强调 MC 算法与其**积分期望形式**的一一对应：

- 便于从积分形式推导新算法（如参数灵敏度 ∂T/∂π）
- **滤波函数（filtering function）** 抽象：光线投射循环与物理处理分离
  - 每次体素相交调用 filter，处理零碰撞拒绝、灵敏度权重 σ 等
  - 与计算机图形学中透明表面 filter 概念一致



### 2.4 预期性能特征

- 表面复杂度：MC 成本对三角形数量**几乎不敏感**（Embree BVH）
- 体数据分辨率：在合并八叉树下，渲染时间对原始网格分辨率**几乎不敏感**
- MC 天然优势：增加积分维度（波长、角度等）成本近似线性增长

---



## 3. 技术途径 / 解决方案

```
┌─────────────────────────────────────────────────────────────┐
│                    应用层 (htrdr 渲染器等)                    │
│   后向路径追踪 · 局部估计法 · 分解追踪 · k-distribution      │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│              应用导向模块 (sky / ground / camera / sun)       │
│   构建大气体八叉树 · 相机/太阳模型 · 地表 OBJ 三角网格        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│           数据导向模块 (htcp / htmie / htgop)                 │
│   LES 云场 · Mie 云滴光学 · RRTMG 气体 k-distribution        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│     底层 MC 库 (Star-Engine: ssp / s3d / svx / ssf)         │
│   随机采样 · 表面光线追踪(Embree) · 体八叉树 · 散射函数        │
└─────────────────────────────────────────────────────────────┘
```

**技术路线总结：**

1. **零碰撞 + 八叉树**：绕过 Beer 非线性，实现体数据 path-tracing 加速
2. **Out-of-core 存储**：原始数据与加速网格存磁盘，Morton 序 + 固定块按需加载
3. **模块化 C 库**：提供底层积木而非单一应用，支持辐射传输专家自定义 MC 算法
4. **Bridging CG ↔ 大气科学**：引入 Woodcock tracking、Disney decomposition tracking、Embree 等图形学成熟技术

---



## 4. 算法软件构成



### 4.0 架构

三层模块体系（见 Appendix C）：


| 层级        | 类型       | 代表模块                                                                                 | 分发方式                                                     |
| --------- | -------- | ------------------------------------------------------------------------------------ | -------------------------------------------------------- |
| **L1 底层** | 通用 MC 抽象 | Star-Sampling (ssp), Star-3D (s3d), Star-VoXel (svx), Star-ScatteringFunctions (ssf) | [star-engine](https://gitlab.com/meso-star/star-engine/) |
| **L2 数据** | 大气专用数据加载 | htcp, htmie, htgop                                                                   | High-Tune 项目内                                            |
| **L3 应用** | 场景/传感器模型 | htrdr sky/ground/camera/sun                                                          | htrdr 渲染应用                                               |


**核心抽象：光线投射 = sophisticated do-while 循环**

```
while (ray active):
    intersect next voxel/surface
    → call filter function (physicist-defined)
    → filter returns: continue / terminate / scatter / reject null-collision
```

**数据流（渲染应用）：**

```
LES NetCDF (T,P,q_l,q_v)
        ↓ htcp
中间 3D 光学系数网格 (per quadrature point, out-of-core)
        ↓ htrdr sky + svx
八叉树加速结构 (VTK, ~7.4 GB for 38 GB raw)
        ↓ htrdr (backward path tracing)
合成云辐射图像 (1280×720 RGB)
```



### 4.1 输入 / 输出 / 接口



#### 输入


| 数据类型      | 来源/格式                            | 加载模块         | 说明                              |
| --------- | -------------------------------- | ------------ | ------------------------------- |
| **3D 云场** | Meso-NH LES, NetCDF              | htcp         | 液态/气态水、温度、压力                    |
| **云滴光学**  | Mie LUT, NetCDF                  | htmie        | 单次散射反照率、消光、不对称因子、相函数            |
| **气体光学**  | RRTMG k-distribution (via ecRad) | htgop        | 16 LW + 14 SW 谱段，各 quadrature 点 |
| **地表**    | OBJ 三角网格                         | htrdr ground | 可含复杂地形 (2×2048×2048 三角形)        |
| **晴空柱**   | I3RC cumulus 剖面 (ecRad)          | htgop        | 1D 垂直廓线                         |
| **场景配置**  | starter-pack 脚本/参数               | —            | 相机、太阳天顶角/方位角、FOV                |




#### 输出


| 输出类型       | 说明                              |
| ---------- | ------------------------------- |
| **合成辐射图像** | 1280×720 像素，RGB（人眼三色锥细胞响应）      |
| **辐射度量**   | 透过率 T、地面通量、参数灵敏度 ∂T/∂α 等（依算法而定） |
| **加速结构**   | 八叉树 VTK 文件（可预计算复用）              |
| **诊断量**    | 路径权重、MC 统计不确定度 (±3σ)            |




#### 关键 API 示例

```c
// 体八叉树
svx_octree_create(...);
svx_tree_trace_ray(...);
svx_hit_filter_T(...);          // 滤波函数类型

// 表面 (Embree)
s3d_scene_create(...);
s3d_scene_view_trace_ray(...);
s3d_hit_filter_function_T(...);

// 大气体
htrdr_sky_create(...);
htrdr_sky_trace_ray(...);
htrdr_sky_fetch_raw_property(...);
htrdr_sky_fetch_svx_property(...);

// 随机采样
ssp_rng_canonical(...);
ssp_ran_exp_pdf(...);           // Beer 指数分布
ssp_ran_hemisphere_cos(...);

// 散射
ssf_phase_sample(...);
ssf_specular_reflection_setup(...);
```

完整函数列表见论文 Table C2 及 [htrdr 注释源码](https://www.meso-star.com/projects/high-tune/high-tune.html)。

### 4.2 模块功能



#### L1 — Star-Engine（底层库）


| 模块                                 | 功能                                            |
| ---------------------------------- | --------------------------------------------- |
| **ssp** (Star-Sampling)            | 可并行伪随机数；指数、半球余弦等 PDF 采样与求值                    |
| **s3d** (Star-3D)                  | 三角网格场景定义；基于 **Embree** 的 BVH 光线追踪与 hit filter |
| **svx** (Star-VoXel)               | 体素八叉树构建与穿越；Morton 序存储；out-of-core 块加载         |
| **ssf** (Star-ScatteringFunctions) | 表面/体散射函数设置、采样与求值（HG、Rayleigh、Lambert 等）       |




#### L2 — 数据导向库


| 模块        | 功能                                |
| --------- | --------------------------------- |
| **htcp**  | 加载 LES 四维大气场（les2htcp 转换工具）       |
| **htmie** | 云滴 Mie 光学查找表：消光、吸收、散射截面、相函数       |
| **htgop** | 气体 RRTMG k-distribution 查找表，含水汽插值 |




#### L3 — 应用导向模块（htrdr）


| 模块               | 功能                                        |
| ---------------- | ----------------------------------------- |
| **htrdr sky**    | 合并 3D 云场 + 1D 气体廓线 → 中间光学网格 → 八叉树；零碰撞射线追踪 |
| **htrdr ground** | OBJ 地表加载、加速结构、反射追踪                        |
| **htrdr camera** | 针孔相机模型，从相机发出后向路径                          |
| **htrdr sun**    | 太阳锥采样、太阳辐亮度访问                             |




#### 应用 — htrdr 渲染器

- **后向** MC 路径追踪（从相机出发）
- 局部估计法（local estimate）更新路径权重
- 分解追踪（decomposition tracking）降低方差
- k-distribution 光谱积分（可见光 30 个 quadrature 点）
- 支持多线程并行（论文实验：40 线程）



### 4.3 运行环境 / 中间件


| 组件                          | 说明                                                                   |
| --------------------------- | -------------------------------------------------------------------- |
| **语言**                      | C（CPU 实现，论文发表时无 GPU）                                                 |
| **Embree**                  | Intel 光线追踪库，表面 BVH 加速 ([Wald et al., 2014](https://www.embree.org/)) |
| **Meso-Star / Star-Engine** | 开发环境与底层库托管平台                                                         |
| **操作系统**                    | Linux（out-of-core 依赖 OS 页缓存与块调度）                                     |
| **外部物理工具**                  | Meso-NH (LES)、ecRad/RRTMG (气体光学)、Mie 散射 Fortran 代码                   |
| **数据格式**                    | NetCDF（场数据与 LUT）、VTK（八叉树）、OBJ（地表）                                    |
| **并行**                      | 多线程（像素/路径分发）；八叉树构建可并行（每 quadrature 点一棵树）                             |
| **内存策略**                    | Out-of-core：原始数据 + 加速网格按需加载；Morton 序 + 固定大小内存块                       |


**资源链接：**

- 底层库：[https://gitlab.com/meso-star/star-engine/](https://gitlab.com/meso-star/star-engine/)
- High-Tune 项目：[https://www.meso-star.com/projects/high-tune/high-tune.html](https://www.meso-star.com/projects/high-tune/high-tune.html)
- Starter-pack（数据 + 复现脚本）：[https://www.meso-star.com/projects/high-tune/starter-pack.html](https://www.meso-star.com/projects/high-tune/starter-pack.html)

**当前局限（Section 6）：**

- 每个 quadrature 点独立八叉树（30 棵），内存与构建成本较高
- Out-of-core 尚未在全库范围实现（需 bundle tracing 等策略）
- 尚未支持冰晶、气溶胶、滴谱空间变化等扩展场
- 灵敏度估计方差较大，收敛加速仍在研究中

---



## 5. 实验



### 5.1 渲染性能：体数据分辨率不敏感性（Section 4.2）

**设置：** Congestus 云场（Meso-NH, 5 m, 1000³），从 5 m 粗化至 200 m，相机/太阳固定，1280×720 × 4096 paths × 3 通道。

**结果（Figure 5）：**

- 合并八叉树（τ̃ = 1）：**渲染时间几乎恒定**，与原始网格分辨率无关
- 不合并（τ̃ = 0）：渲染时间随分辨率**指数增长**
- 38 GB 原始数据 → 7.4 GB 八叉树（VTK）



### 5.2 多云型对比（Section 4.3）


| 场景           | 网格规模        | 平均路径时间 t [µs] | 总渲染时间 (40 线程) |
| ------------ | ----------- | ------------- | ------------- |
| Congestus 5m | 1000³       | 117.99        | 9h 22min      |
| BOMEX        | 512×512×160 | 37.26         | 2h 59min      |
| ARMCu 1      | 256×256×160 | 105.05        | 8h 22min      |
| ARMCu 2      | 256×256×160 | 60.43         | 4h 59min      |
| FIRE         | 250×250×70  | 122.06        | 10h 01min     |


**结论：**

- 表面复杂度（2048² 三角形 vs 2 三角形）**不影响**渲染时间
- 时间差异主要来自：相机视场中云像素比例、云光学厚度、多次散射路径长度
- 最优合并判据 τ̃ 约在 **1–10** 之间（Figure 7）



### 5.3 表面复杂度不敏感性（Figure 1）

- 地形三角形从少到 2×2048×2048，渲染时间相对最精细场景变化极小
- 与 Embree BVH 加速一致



### 5.4 参数灵敏度（Section 5.1, Figure 8）

- **指标：** 地面单色通量透过率 T 及其对云滴吸收比 α 的灵敏度 ∂T/∂α
- **方法：** 积分形式求导 → filter 中累积 σ 权重（Equation 3）
- **发现：** 云边 3D 泄漏使晴空 T > 1；灵敏度在云下最大；灵敏度方差显著大于 T 本身



### 5.5 大尺度参数化评估（Section 5.2, Figure 9）

本节直接检验 §1.4 所述简化方案的系统偏差：以精确 MC 为基准，对比 ecRad 二流求解器及其 3D 扩展。

**背景（与 §1.4.3 衔接）：** TripleClouds、SPARTACUS 等 ecRad 求解器基于**二流近似**传播漫射通量，并常用 **delta-Eddington 缩放**修正云滴前向散射——减小等效 τ、降低等效 ω，使总通量接近正确，但 **direct / diffuse 划分可能仍偏**。二流模型仅用两个斜向传播 diffuse，也会系统性低估大天顶角下的直射衰减。

**实验设置：**

- **场景：** ARMCu 积云 LES 场（第 8 小时，1530 LT）
- **指标：** 地面水平平均 **direct / total** 宽带太阳通量比，随太阳天顶角变化
- **MC 参考：** 前向 MC，RRTMG k-distribution 光谱采样；真实 **Mie 相函数**
- **对比组：**
  - ecRad **TripleClouds**（1D 二流 + 子网格云量三重划分）
  - ecRad **SPARTACUS**（TripleClouds + 水平传输的 3D 效应）
  - MC + **delta-Eddington 缩放 HG** 相函数（模拟参数化中的相函数近似）

**结果：**

| 对比 | 发现 |
|------|------|
| TripleClouds vs MC | 大天顶角下 **direct 分量偏高**——未体现云缘拦截导致的 direct 衰减 |
| SPARTACUS vs MC | 引入 3D 水平传输后 **与 MC 吻合良好**，有效修正天顶角依赖 |
| MC (Mie) vs MC (δ-HG) | δ-Eddington 缩放 HG **高估 direct 通量**——等效 τ 被削减的副作用 |

**物理解读：** 天顶角增大 → 有效云量增大 → 更多直射光经云**侧缘**入射并被散射。纯 1D 方案无法表达此 3D 几何效应；SPARTACUS 通过参数化侧向传输部分弥补。δ-Eddington 虽改善总通量，却在 direct/diffuse 分拆上引入**结构性偏差**——这正是精确路径追踪作为验证工具的价值所在。



### 5.6 基准验证

- 与 Galtier et al. (2013) Table 1 多重散射基准一致（MC 统计误差内）
- 与 3DMCPOL / IPRT cubic cloud case 一致（Supporting Information）

---



## 参考文献

- Villefranque N. et al. (2019). *Path-tracing Monte Carlo Library for 3D Radiative Transfer in Highly Resolved Cloudy Atmospheres*. JAMES. [arXiv:1902.01137](https://arxiv.org/abs/1902.01137)
- 项目主页：[https://www.meso-star.com/projects/high-tune/high-tune.html](https://www.meso-star.com/projects/high-tune/high-tune.html)
- Star-Engine：[https://gitlab.com/meso-star/star-engine/](https://gitlab.com/meso-star/star-engine/)


# 云辐射仿真端到端流程

> **范围：** 本仓库 `cloudGS` 中，以 **后向 Monte Carlo 路径追踪** 求解三维大气中云与气体的辐射传输，并生成短波（及可选长波）图像或对照量。  
> **准绳实现：** Meso-Star **htrdr** 大气求解器（见 [`methods.md`](methods.md)）+ Atmosphere Starter Pack 云/气/太阳数据。  
> **本仓库 GPU 实现：** `gpu/threedgptir_tracer/`（OptiX），主机侧资产管线在 `tools/`（如 `mountain_gs_scene.py`、`fire_scene.py`、`render_*_gpu.py`）。差异见 [`diff.md`](diff.md)。  
> **单位：** 长度 m；温度 K；短波 CIE 像素为通道刺激值对应的辐亮度量纲 W·sr⁻¹·m⁻²；单色谱密度为 W·sr⁻¹·m⁻²·m⁻¹。  
> **本文不造词：** 术语取辐射传输与图形学惯例（参与介质、消光、光学厚度、自由程、零碰撞 / Woodcock、majorant、相函数、次事件估计等）。

---

## 0. 读本文的路线图

```text
物理问题 (RTE)
  → 需要哪些场与边界 (LES 云、气体、太阳、地面)
    → 如何把微物理变成光学系数 (Mie × 液水含量)
      → 如何把光学场装进加速结构 (体素砖 / 八叉树 / 1D 气层)
        → 一条路径如何无偏采样 (光谱 → 自由程 → 碰撞类型 → 散射/NEE)
          → 像素如何估计 (CIE / 单色 / 长波)
            → 本仓库脚本如何把上述串成可跑实验
              → 与「体积 3DGS 替代云体素」计划的边界 (另文)
```

---

## 1. 物理问题：三维大气中的辐射传输方程

### 1.1 参与介质

云滴与大气分子构成 **参与介质（participating medium）**：辐射在传播中被 **吸收（absorption）** 与 **散射（scattering）**。位置 \(\mathbf{x}\) 上常用光学系数：

| 符号 | 名称 | 含义 |
|------|------|------|
| \(k_a(\mathbf{x},\nu)\) | 吸收系数 | 单位长度上吸收的概率密度（m⁻¹） |
| \(k_s(\mathbf{x},\nu)\) | 散射系数 | 单位长度上散射的概率密度（m⁻¹） |
| \(k_e=k_a+k_s\) | 消光系数（extinction） | 吸收与散射之和 |
| \(\omega=k_s/k_e\) | 单次散射反照率（single-scattering albedo） | 碰撞为散射的概率（\(k_e>0\)） |
| \(p(\boldsymbol{\omega}'\!\leftarrow\!\boldsymbol{\omega})\) | 相函数（phase function） | 散射方向的概率密度（对立体角归一） |

沿射线方向 \(\boldsymbol{\omega}\)，**光学厚度（optical depth）**

\[
\tau(s)=\int_0^s k_e\bigl(\mathbf{x}_0+t\boldsymbol{\omega}\bigr)\,dt.
\]

**Beer–Lambert 透过率** \(\mathrm{Tr}=\mathrm{e}^{-\tau}\)。辐亮度满足辐射传输方程（RTE）：沿路径的衰减、散射内源、（长波）热发射等源项耦合，一般无解析闭式解。

### 1.2 为何用后向路径追踪

前向从太阳撒光子再收集到传感器，在高光学厚度云中效率差。本管线采用 **后向 Monte Carlo**：从传感器发出射线，在介质与地面上随机游走，用 **无偏估计量** 逼近观测算子（像素辐亮度、CIE 刺激值等）。期望等于目标积分；样本数（spp）只降低方差，不改变期望（在实现保持无偏的前提下）。

### 1.3 本问题中的「云仿真」指什么

在本仓库语境下，**云仿真**不是流体 LES 本身，而是：

1. 给定 LES 输出的云微物理/热力学场（及气体廓线、太阳、地面）；
2. 映射为光学场 \(k_a,k_s\) 与相函数参数；
3. 用路径追踪求解传感器观测。

LES 负责「云在哪里、含多少液水」；辐射求解器负责「给定光学场，相机看到什么」。

---

## 2. 输入数据层：场景由哪些物理与几何要素构成

### 2.1 云场：HTCP（LES 体数据）

Starter Pack 与本仓库脚本使用 **HTCP** 格式的云盒子，典型场包括：

| 量 | 含义 |
|----|------|
| `RCT` | 云滴液态水混合比（相对干空气，kg/kg） |
| `T` | 温度（K） |
| `PABST` | 气压（Pa） |
| `RVT` | 水汽混合比（用于云内气体 \(x_{\mathrm{H_2O}}\) 插值） |

水平分辨与垂直分层因个例而异，例如：

| 个例 | 典型用途 | 水平 | 垂直特征 |
|------|----------|------|----------|
| ARMCu `DZVAR...htcp` | `mountain` 大陆积云 | 25 m，约 6.4 km 盒 | 不规则 Z 层 |
| BOMEX `L12km...htcp` | 海洋积云 / 通量 | 25 m 级 | 云层有限高度 |
| FIRE `L25.1.FIRE.012.htcp` | `fire_gs` 层积云 | 100 m，约 25 km 盒 | 薄云层 |

**Z 不周期**；XY 是否周期由求解器选项决定（htrdr `-r`；GPU 侧另有实现取舍，见 [`diff.md`](diff.md)）。

### 2.2 云滴光学：Mie 查找表（htmie）

液水质量浓度 \(q_l\) 乘以 **质量吸收/散射截面** 得粒子光学系数。截面来自预计算的 Lorenz–Mie 查找表（如 `Mie_LUT_Cloud-2-10-0.010.nc`：对数正态滴谱，有效半径与有效方差由 LUT 约定固定）。

运行时短波体积相函数在本管线中取 **Henyey–Greenstein（HG）**，不对称因子 \(g(\lambda)\) 来自 Mie LUT 的波长依赖；**完整 Mie 相函数表通常不直接用于短波体积采样**（与 2019 论文部分实验设定不同，见 [`methods.md`](methods.md) 开篇说明）。

### 2.3 气体：一维垂直层结（htgop / ecRad 相关 \(k\) 分布）

气体吸收与（短波）Rayleigh 散射由 **1D 垂直廓线** 给出：高度层 + RRTMG 风格 **短波 / 长波谱段** + 每谱段若干 **quadrature（\(k\) 点）** 及权重。水平方向均匀（忽略地球曲率），云盒子内可用 LES 水汽修正该层 \(k\)。

气体散射相函数为 **Rayleigh**：

\[
p_{\mathrm{R}}(\mu)=\frac{3}{16\pi}(1+\mu^{2}),\quad \mu=\boldsymbol{\omega}_{\mathrm{in}}\cdot\boldsymbol{\omega}_{\mathrm{out}}.
\]

长波气体散射系数常取 0（无分子散射贡献）。

### 2.4 太阳：解析光源（非网格）

短波外源为 **有限立体角太阳盘**：

- 半张角 \(\theta_\odot\approx 4.6524\times 10^{-3}\,\mathrm{rad}\)；
- 立体角 \(\Omega_\odot=2\pi(1-\cos\theta_\odot)\)；
- 方向由方位角 / 高度角给出；
- 单色辐亮度 \(L_\odot(\lambda)=B_\lambda(T_\odot)\)，\(T_\odot=5778\,\mathrm{K}\)（Planck），**不是**谱段平均太阳常数贴一张图。

没有 HDRI / 天空球作为短波边界：太空 miss 为黑；晴空蓝色来自气体多次散射后对太阳的 **次事件估计（next-event estimation, NEE）**。

### 2.5 地面与其它表面

| 类型 | 作用 | 本仓库常见形态 |
|------|------|----------------|
| 三角网格 | 山、沙漠、海面等 | OBJ → Embree（htrdr）或 OptiX GAS（GPU）；可 XY 周期 slab |
| 3D Gaussian（表面） | 可选山地表示 | `--ground gs`：Bernoulli \(\alpha\) + BRDF，**不贡献云的 \(k\)** |
| 材料 | Lambert / 镜面等 | MruMtl 光谱节点；短波用反射，长波另用温度发射 |

地面与云体积在路径上 **竞争最近命中**：先比较自由程事件距离与表面交点距离。

### 2.6 传感器

| 类型 | 作用 |
|------|------|
| 针孔透视 | 主图像模式 |
| 正交相机 | 俯视 / 对照实验 |
| （htrdr）薄透镜、通量图 | GPU 主图像管线通常未接 |

每个像素多个独立路径样本（spp）；CIE 模式下常对 X/Y/Z 通道分别采样波长。

---

## 3. 光学构建：从液水到 \(k_a,k_s,g\)

### 3.1 液水质量浓度

干空气密度与液水含量（与 [`methods.md`](methods.md) §6.1 一致）：

\[
\rho_{\mathrm{da}}=\frac{P\,M_{\mathrm{air}}}{T\,R},\qquad
q_l=\rho_{\mathrm{da}}\cdot\mathrm{RCT}.
\]

### 3.2 粒子光学系数

对当前光谱条件取截面 \(\overline{C}_a,\overline{C}_s\)（htrdr：常为 **整个 RRTMG 谱段** 上 Mie 截面的线性平均）：

\[
k_{a,\mathrm{p}}=\overline{C}_a\,q_l,\qquad
k_{s,\mathrm{p}}=\overline{C}_s\,q_l.
\]

不对称因子 \(g=g(\lambda)\) 供 HG 使用（按样本波长或路径绑定更新）。

### 3.3 本仓库 GPU 的烘焙与缩放

主机 `load_htcp_voxels`（`tools/mountain_gs_scene.py`）典型流程：

1. 读 HTCP → 计算各单元 \(q_l\)；
2. 用参考波长（常 **550 nm**）Mie 截面写入规则网格上的 `ka`、`ks`、`temp`（及可选 `x_h2o`）；
3. 序列化为 `*.bin`（如 `fire_voxels.bin`）供 OptiX 上传。

路径上再乘谱段缩放（概念上）：

\[
k_s\leftarrow k_s\cdot\frac{C_s(\mathrm{band})}{C_s(\mathrm{ref})},\qquad
k_a\leftarrow k_a\cdot\frac{C_a(\mathrm{band})}{C_a(\mathrm{ref})},
\]

并绑定当前 \(g\)（如 `mie_g`）。这与 htrdr「每谱段直接用该段 \(\overline{C}\) 乘 \(q_l\)」期望接近，实现路径不同（见 [`diff.md`](diff.md)）。

### 3.4 规则网格 vs 不规则 Z

- **htrdr：** 可保留 HTCP 不规则垂直层，查询时做层映射；云加速为 **八叉树（SVX）**，叶内存 \(k_{\min},k_{\max}\)。  
- **GPU `AtmVoxelGrid`：** ABI 为 **均匀 \(\mathrm{d}x,\mathrm{d}y,\mathrm{d}z\)** 的 `ka/ks/temperature` 数组 + 每砖 `kext_max`（短波游走常用砖内 \(\max k_s\)）。不规则 Z 在加载时重采样/摊平，引入表示误差。

---

## 4. 场景装配：可运行实验的资产图

### 4.1 逻辑组件图

```text
                    ┌─────────────────┐
                    │ 传感器 (相机)    │
                    └────────┬────────┘
                             │ 主射线
         ┌───────────────────▼───────────────────┐
         │     路径积分器 (SW / LW / CIE)          │
         └─┬─────────────┬─────────────┬─────────┘
           │             │             │
           ▼             ▼             ▼
    ┌────────────┐ ┌──────────┐ ┌──────────────┐
    │ 云体素场    │ │ 1D 气体   │ │ 地面/GS/网格  │
    │ ka,ks,T    │ │ 层结 k(z) │ │ BRDF / 几何   │
    └─────┬──────┘ └────┬─────┘ └──────┬───────┘
          │             │              │
          ▼             ▼              │
    ┌────────────┐ ┌──────────┐        │
    │ HTCP+Mie   │ │ htgop    │        │
    └────────────┘ └──────────┘        │
                                       ▼
                              ┌────────────────┐
                              │ 太阳盘 (解析)   │
                              └────────────────┘
```

### 4.2 本仓库典型场景入口

| 场景目录 / 脚本 | 云 | 地面 | 备注 |
|-----------------|----|------|------|
| `scenes/mountain_gs` + `render_mountain_gs_gpu.py` | ARMCu 类体素 | mesh 或表面 GS | 与 `mountain.sh` 对照主战场 |
| `scenes/fire_gs` + `fire_scene.py` / `render_fire_*` | FIRE 层积云 | 海面网格等 | 大水平域；SWIR/MWIR/LWIR 实验 |
| htrdr Starter Pack 脚本 | 同上数据源 | OBJ | CPU 准绳 |

主机准备示例（FIRE）：扩展/写出 `fire_voxels.bin`、`fire_voxels.mie.json`、海面网格与相机/太阳参数，再交 GPU 追踪器。

### 4.3 图元职责（不可混淆）

| 图元 | 在传输中的角色 |
|------|----------------|
| 云体素 | **参与介质**：提供 \(k_a,k_s\)，Woodcock 自由程，HG 散射 |
| 气体层 | **参与介质**：1D \(k(z)\)，Rayleigh（SW），与云 **竞争风险** |
| 三角地面 / 表面 GS | **表面**：BSDF / BRDF；遮挡；短波反射太阳；长波可发射 |
| 太阳 | **外源**：直射与 NEE |

表面 3DGS 的 \(\alpha\) **不是**云的消光场；大气核中 `atm_gs_closest_surface` 将随机 \(\alpha\) 视为物体表面模型（见计划文档与路径追踪注释）。

---

## 5. 核心算法层：无偏路径积分如何推进

### 5.1 光谱样本如何产生（短波 CIE 图像）

目标是通道积分 \(\int L(\lambda)\bar{c}(\lambda)\,d\lambda\)。无偏做法：

1. 按配色函数诱导的密度 \(p_C(\lambda)\) 抽取波长 \(\lambda\)；
2. 映射到 RRTMG **谱段** `iband` 与 **quadrature** `iquad`；
3. 绑定该路径的气体 \(k\)、云截面缩放、\(g\)、太阳 Planck \(L_\odot(\lambda)\)；
4. 在此 **固定光谱状态** 下估计 \(L\)，再除以 \(p_C\)（实现里注意 nm⁻¹ 与 m⁻¹ 单位换算）。

单色或长波模式则按各自谱密度（如 Planck 参考温度）抽 \(\lambda\) / 谱段。

### 5.2 自由程：Woodcock 追踪（零碰撞 / 最大截面法）

非均匀 \(k(\mathbf{x})\) 上，光学厚度首次到达服从单位指数分布。给定区域 **majorant** \(\hat{k}\ge k\)：

1. 抽取 \(\tau\sim\mathrm{Exp}(1)\)；
2. 候选步长 \(\Delta s=\tau/\hat{k}\)（或在叶/砖内消耗光学厚度预算）；
3. 以概率 \(k/\hat{k}\) **接受真实碰撞**，否则记 **零碰撞（null collision）** 并继续——等价于前向 \(\delta\) 散射，期望无偏。

**短波散射游走** 使用的系数往往是 **散射系数场** \(K_s\)（气体 \(k_s\) + 粒子 \(k_s\)），吸收透过率另用模拟估计；**长波** 常用消光 \(K_e\) 游走，碰撞后再以 \(\omega\) Bernoulli 分为吸收/散射。细节与 htrdr 分解追踪见 [`methods.md`](methods.md) §2。

### 5.3 Majorant 在本仓库 GPU 上的形态

对规则云网格，按砖（默认边长 8 体素）预计算：

- 短波：\(\hat{k}=\max_{\mathrm{brick}} k_s\)（与 htrdr SW 叶 majorant 角色同类）；
- 长波：常用 \(\max(k_a+k_s)\)。

气体则在层上取该层（及 quadrature）可用的上界并沿 Z 推进。GPU 为 **气、云两路 Woodcock 再取较早事件（竞争风险）**；htrdr 文档主干是合并 \(K_s\) 上一次追踪——泊松首次事件期望可对齐，majorant 松紧与实现细节不同（[`diff.md`](diff.md) §2.2）。

### 5.4 碰撞之后：散射、吸收、相函数

| 事件 | 行为 |
|------|------|
| 粒子散射 | 按 HG(\(g\)) 抽样新方向；吞吐乘单次散射相关权重（依估计量形式） |
| 气体散射 | Rayleigh 抽样新方向 |
| 吸收（LW 等） | 局部估计 Planck \(B_\lambda(T)\) 并终止，或按模式累积 |
| 表面命中 | BSDF 采样或 NEE；短波表面俄罗斯轮盘用反照率相关存活概率 |

气 vs 粒：在合并解释下，以 \(k_{s,\mathrm{gas}}/(k_{s,\mathrm{gas}}+k_{s,\mathrm{particle}})\) 选择 Rayleigh 或 HG（只掷一次，勿重复乘因子）。

### 5.5 次事件估计（NEE）与透过率

短波在体积或表面碰撞点对太阳盘方向做 **局部估计**：

- 在太阳锥内抽样方向（或检验方向是否在锥内）；
- 乘相函数或 BRDF×余弦；
- 乘太阳辐亮度与 **消光透过率** \(\mathrm{Tr}_{\mathrm{ext}}\)（沿段积分 \(k_e\) 的 \(\mathrm{e}^{-\tau}\) 或其无偏估计）；
- 地面遮挡则透过率为 0。

htrdr 对 \(\mathrm{Tr}\) 使用 **分解追踪（analog 下界 + 残差 delta-tracking）**；GPU 侧多为 DDA / 累积光学厚度的模拟透过率（实现差异见 [`diff.md`](diff.md)）。

相机射线若本身落在太阳锥内，可另计 **直射** 日盘贡献（同样乘透过率）。

### 5.6 与表面的耦合

每步（或每事件）在体积候选距离与 **最近表面交点** 之间取较小者：

- 表面更近 → 执行 BRDF / 发射逻辑；
- 体积更近 → 执行介质事件。

GPU 网格地面支持周期 slab（最多有限格步）；表面 GS 用随机 \(\alpha\) 接受最近高斯命中。体积 NEE 遮挡策略需避免错误的「向上太阳被山 GS 误挡」类失效模式（工程注释与工作区规则约束地面路径改动）。

### 5.7 路径终止

- 逃出大气上界 / miss 且无源；
- 吸收；
- 表面俄罗斯轮盘失败；
- （GPU）达到 `max_bounces` 等实现上限。

无偏性要求：终止规则要么覆盖测度为零的集合，要么用俄罗斯轮盘等 **逆概率加权** 补偿；硬截断吞吐而不补偿则引入偏差。

---

## 6. 一条短波 CIE 像素的逐步时间线

下列顺序与 [`methods.md`](methods.md) §8 及 GPU `atmospherePathTracer` 意图一致，便于对照实现：

```text
1. 选像素、抖动胶片点 → 生成相机射线 (o, ω)
2. 抽 CIE 通道波长 λ → 定 iband, iquad → 绑定气体 k、云缩放、g、L_☉(λ)
3. 吞吐 ξ ← 1；累积辐射 L ← 0
4. 若 ω 在太阳锥内：L += L_☉ · Tr_ext(相机段)   [直射]
5. loop:
   a. 在气体与云上做自由程采样（Woodcock / 竞争风险），得候选体积事件距离 t_vol
   b. 求地面/GS 最近表面距离 t_surf
   c. 若二者皆无：break（太空黑）
   d. 若 t_surf < t_vol：表面交互
        - NEE 太阳（BRDF × Tr）
        - BSDF 采样新方向；表面 RR；更新 ξ
   e. 否则：体积事件
        - 判气/粒；吸收则估源并停，或散射
        - 体积 NEE：相函数 × L_☉ × Tr_ext / p_☉
        - 抽样新方向；乘吸收透过等权重（依 SW 估计量）
6. 将 L 按光谱 pdf 变回通道贡献；与其它 spp / 通道平均
```

长波去掉太阳，改为消光游走 + 吸收/表面处的 Planck 局部估计；上界 miss 为 0。

---

## 7. 实现栈与数据落盘

### 7.1 分层

| 层 | 位置 | 职责 |
|----|------|------|
| 论文与 htrdr 方法 | [`docs/README.md`](README.md)、[`docs/methods.md`](methods.md) | 物理与 CPU 准绳算法 |
| GPU 差异 | [`docs/diff.md`](diff.md)、[`docs/ir_methods.md`](ir_methods.md) | 与准绳的偏差清单（含红外） |
| 主机资产 | `tools/*_scene.py`、`load_htcp_voxels`、`load_voxel_bin` | HTCP→`ka/ks`、写 bin、相机太阳 |
| 设备积分器 | `atmosphere.cuh`、`atmospherePathTracer.cuh` | Woodcock、NEE、表面、光谱绑定 |
| 表面 GS / PTIR | `gaussianParticles.cuh`、`pathTracer.cuh` | 非大气或地面 GS；α 合成 / 路径表面 |
| 对照实验 | `tools/render_*`、`tools/compare_*` | spp、CIE、与 htrdr 图/指标 |

### 7.2 云体素二进制（概念布局）

`load_voxel_bin` 读入：网格尺寸、原点、间距、`ks`/`ka`/`temp` 数组（及可选 `z_iface`、`x_h2o`）。上传为 `AtmVoxelGrid` 后由 `build_brick_majorants` 填砖 majorant。

### 7.3 数值与精度

- htrdr 大量 **double**；GPU 大气路径多为 **float**。  
- 大场景（如 FIRE 百万～千万级活跃体素）影响 majorant 表大小、内存与方差。  
- OptiX 内核变更需清理磁盘缓存并提升内核修订号，避免静默使用旧模块。

---

## 8. 验证与「对齐」指什么

### 8.1 对照层级

| 层级 | 内容 | 意义 |
|------|------|------|
| 物理配置 | 同一 HTCP、太阳几何、相机、材料 | 可比 |
| 估计量 | 同一 SW/LW/CIE 定义 | 期望应对准 |
| 实现 | 网格规则化、气云竞争、Tr 估计、浮点 | 允许已知偏差 |
| 指标 | 像素均值、relRMSE、分位、剖面 | 定量，禁止仅目视 |

[`diff.md`](diff.md) 列出 GPU 相对 `methods.md` 的已知差异；宣称「与 htrdr 一致」必须落在已声明可接受的差异集合内。

### 8.2 无偏 vs 低方差

- **无偏：** 估计量期望等于目标积分（在模型与采样设计下）。  
- **低方差：** spp、重要性采样、紧 majorant、谱重要性等。  

改相函数（HG↔完整 Mie）、给 miss 垫色、硬截断路径吞吐而不补偿等，会 **改变偏差**，不能用「多采样」消除。

---

## 9. 与「体积高斯替代云体素」的边界（现状）

当前生产路径中：

- **云 = 体素参与介质 + Woodcock；**  
- **山可选 = 表面高斯；**  
- **气 = 1D 层结。**

将云改为体积 3DGS（\(\sigma(\mathbf{x})=\sum_i\sigma_i G_i(\mathbf{x})\)、砖 majorant、同一大气 MC 算子）属于 **进行中的设计**，见 [`plan_gs_replace_all_primitives.md`](plan_gs_replace_all_primitives.md)。其中：

- 主机 P1 探针工具：`tools/volume_gs_p1.py`（σ 查询与 majorant 合法性；**未**接入大气核）；  
- 主机 P1b 可微对齐：`tools/volume_gs_fit_torch.py`（中心锁死、Adam 学 scale/quat/振幅，网格 MSE vs pin；**未**接入大气核）；  
- 不得把表面 \(\alpha\) 合成或材质字段改名冒充云多次散射；  
- majorant 必须保持 \(\hat{k}\ge k\)（或 \(\bar\sigma_b\ge\sigma(x)\)）；对核的加速查询可以裁剪，**写入上界时不得无补偿地丢掉仍贡献的尾项**。

本文第 1–8 节描述的是 **已落地的体素云仿真全流程**；体积 GS 是表示层替换提案，不是现状默认路径。

---

## 10. 术语表（本文用法）

| 术语 | 含义 |
|------|------|
| 参与介质 | 沿路径连续发生吸收/散射的三维场 |
| 消光 / 散射 / 吸收系数 | \(k_e,k_s,k_a\)（m⁻¹） |
| 光学厚度 | \(\tau=\int k\,ds\) |
| 透过率 | \(\mathrm{Tr}=\mathrm{e}^{-\tau}\)（或其无偏估计） |
| 自由程 | 至下一次真实碰撞的距离 |
| Woodcock / 零碰撞 / 最大截面法 | 用 majorant 采样自由程，以 \(k/\hat{k}\) 接受真碰 |
| Majorant | 区域上对 \(k\)（或 \(k_s\)）的点态上界 |
| 竞争风险 | 多个独立指数时钟取最先发生者（气 vs 云等） |
| 相函数 | 散射方向分布；本文云侧 HG，气侧 Rayleigh |
| 次事件估计 (NEE) | 在碰撞点显式连接光源的局部估计 |
| 单次散射反照率 | \(\omega=k_s/k_e\) |
| k-distribution / quadrature | 谱段内相关 \(k\) 点离散 |
| HTCP | LES 云场容器格式 |
| Mie 截面 | 由滴谱积分得到的质量截面，用于 \(q_l\mapsto k\) |
| 路径追踪 | 后向随机游走估计光测量算子 |
| 俄罗斯轮盘 | 以概率终止并乘 \(1/q\) 的无偏吞吐补偿 |

---

## 11. 相关文档索引

| 文档 | 内容 |
|------|------|
| [`methods.md`](methods.md) | htrdr 大气路径追踪准绳（逐步、公式） |
| [`diff.md`](diff.md) | GPU 实现相对准绳的差异 |
| [`ir_methods.md`](ir_methods.md) | 红外 / 长波相关方法与对照 |
| [`README.md`](README.md) | 论文背景与瓶颈综述 |
| [`plan_gs_replace_all_primitives.md`](plan_gs_replace_all_primitives.md) | 体积/表面 GS 统一图元计划（非默认云路径） |

---

## 12. 一页纸总结

1. **数据：** LES（HTCP）给液水与温度；Mie 给截面与 \(g\)；htgop 给气体 \(k(z,\nu)\)；太阳为 Planck 盘；地面为网格或表面高斯。  
2. **光学：** \(q_l\mapsto(k_a,k_s)\)；GPU 常先烘焙再谱段缩放。  
3. **结构：** 云 → 规则砖 + majorant（GPU）或八叉树（htrdr）；气 → 1D 层。  
4. **积分：** 光谱重要性采样 → Woodcock 自由程 → 气/粒/表面分支 → HG/Rayleigh/BRDF → 太阳 NEE 与透过率 → 像素平均。  
5. **验证：** 与 htrdr 同配置定量比；清楚列出实现偏差；云必须是参与介质算子，不是表面 \(\alpha\) 合成的别名。

*文档版本：与 `feat/gs` 文档树同步整理；算法细节以 `methods.md` / 源码为准，本文负责端到端层级与职责边界。*

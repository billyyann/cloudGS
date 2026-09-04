# htrdr 大气路径追踪方法

> 准绳：Atmosphere Starter Pack 0.8.0（`htrdr-atmosphere-spk/`）+ 本仓库 `htrdr` 0.12 源码。  
> 主场景：`mountain.sh`。其它脚本只改图元组合与相机/太阳，**同一套求解器**。  
> 单位：长度 1 = 1 m；温度 K；短波 CIE 像素辐亮度 W·sr⁻¹·m⁻²；单色/谱段积分辐亮度密度 W·sr⁻¹·m⁻²·m⁻¹。

本文按源码梳理 htrdr **实际在算什么**，不是论文综述。论文层见同目录 `README.md`。

### 与 Villefranque et al. 2019 的差异

*不能把本文当成论文的逐句复述，也不能用论文 §5.2 的「真 Mie」去要求当前 `mountain.sh`。*

*更准确的关系：库的思想（零碰撞、八叉树、filter、积分↔算法）相符。§4.1 那条短波图像算法主干相符，但 \(g(\lambda)\) 取样点和截面平均方式与 2019 表述有差。§5 其它算法、性能实验、Mie 参考通量本文没覆盖；其中 Mie 还和当前渲染器相反。要对齐论文实验，应对照 Table C1 的 4096 spp、谱段中心 \(g\)、以及 §5.2 的 Mie 相函数，而不是只对照现在的 `mountain.sh`。*

*具体差别如下。*

- *相函数：论文 §4.1 渲染器用 HG，\(g\) 与单次散射反照率取 **RRTMG 谱段中心波长** 的 Mie；当前代码 \(g=g(\lambda)\)，\(\lambda\) 是 CIE 抽到的波长，\(C_a,C_s\) 是 **整个谱段平均**，不是中心波长。论文 §5.2 通量实验另用 **真 Mie 相函数**（以及 δ-Eddington 缩放 HG）作参考；当前 `atmosphere_compute_radiance_sw` 短波体积永远是 HG，LUT 里的 Mie 相函数表不用。*
- *样本数：论文图与 Table C1 为每 CIE 通道 **4096** spp；`mountain.sh` 为 **256** spp。相机/太阳几何与 Table C1「ARMCu 1」相同（方位 225°、天顶角 60° 即高度角 30°、fov 30°），不是另一套场景。*
- *CIE：论文引 Smith & Guild 1931 配色；代码是 Wyman–Sloan–Shirley 2013 解析拟合。*
- *滴谱：论文 Appendix B 写对数正态 \(r_e=10\,\mu\mathrm{m}\)、标准差 1 μm；Starter Pack 0.8.0 写有效方差 0.01（`Mie_LUT_Cloud-2-10-0.010.nc`）。*
- *「30 个 quadrature」：论文指可见光短波 **k 点总数**；本文写 RRTMG **14 个 SW 谱段**（区间数）。不是 14 对 30 的硬冲突。*
- *论文有、本文没有：式 (2)(3) 零碰撞积分形式、§5.1 灵敏度 \(\partial T/\partial\alpha\)、§5.2 对 TripleClouds/SPARTACUS 的前向通量、Congestus \(1000^3\) 性能、out-of-core/Morton。*
- *本文有、2019 论文没有（其后软件）：长波 Planck 路径、`-r/-R` XY 周期、镜面海面、正交相机、通量图、薄透镜、SW 上 \(K_s\) Woodcock 与 \(\mathrm{Tr}(K_a)\) 拆开的实现细节。*

---

## 0. 一句话

后向 Monte Carlo：从传感器发出光线，在 **地面三角网格** 与 **气体+云的参与介质** 上做无偏随机游走；短波唯一外源是 **有限立体角太阳盘**；长波源是 **介质与地面的 Planck 发射**。没有天空球、没有 HDRI、没有环境贴图。晴空蓝色来自气体 Rayleigh 多次散射，不是 miss 上垫的颜色。

主循环在 `htrdr/src/atmosphere/htrdr_atmosphere_compute_radiance_sw.c`（短波）与 `..._lw.c`（长波）。体积场由 High-Tune Sky（`htsky`）提供。

---

## 1. 图元清单

htrdr-atmosphere 只认下面几类图元。Starter Pack 把它们组合成不同脚本，求解器不变。

### 1.1 几何表面（Star-3D / Embree）

| 图元 | 文件 | 规模 / 范围 | 周期 | 材料接口 |
|------|------|-------------|------|----------|
| 山 | `models/mountain.obj` | 8 388 608 三角；AABB `[0,0,0]–[10000,10000,700]` m | XY 可周期（`-R`） | `usemtl air:sand` |
| 沙漠 | `models/desert.obj` | 2 097 152 三角；10 km × 10 km，沙丘高 ≤ 100 m | 同上 | `air` / `desert_sand` |
| 海面 | `models/sea_3m.obj` | 8 388 608 三角；10 km × 10 km，波高 ≤ 3 m | 同上 | `air` / `blue_water` |
| 城市 | `models/city.obj` | 19 517 682 三角，多材料 | 同上 | `city.mtls` + SLUM |
| 平面 | `models/plane.obj` | 四边形，10 m × 10 m | 可选 | `air` / `sand` |

OBJ 是 `htrdr-obj`：每个面写 `usemtl front[:back]`。`air` 是半透明「天空侧」占位（`none`，无 BRDF）；对面才是真实 BSDF。

`mountain.sh`：`-g mountain.obj -R`，地面在 XY 上无限重复。求交走 `htrdr_slab_trace_ray`（最多 32 个周期格），每个格内 Embree BVH。

### 1.2 地面材料（MruMtl）

| 名称 | 文件 | 类型 | 可见光附近反照率 | 温度（仅长波发射） |
|------|------|------|------------------|-------------------|
| `sand` | `materials/legacy/sand.mrumtl` | 分段 Lambert | 300 nm: 0.05；600 nm: 0.16；… | 300 K（`mountain.mtls`） |
| `desert_sand` | `legacy/desert_sand.mrumtl` | Lambert | 光谱分段 | 320 K |
| `blue_water` | `legacy/blue_water.mrumtl` | **镜面** | — | 290 K |
| SLUM 城市材 | `materials/slum/*.mrumtl` | Lambert / 镜面 | 各文件 | 见 `city.mtls` |
| `air` | `none` | 无 BSDF | — | — |

波长处 BRDF：`mrumtl_fetch_brdf(λ, u)` 在光谱节点间掷一次，再实例化 `ssf_lambertian_reflection` 或 `ssf_specular_reflection`。

### 1.3 云场（htcp 体素）

| 文件 | 物理 | 水平 | 垂直 | 用途 |
|------|------|------|------|------|
| `clouds/DZVAR.1.ARMCU.008.diaKCL.htcp` | ARM SGP 大陆积云，LES 第 8 小时 | 6.4 km × 6.4 km，Δx=Δy=25 m，**256×256** | **160 层不规则 Z**（约 0–4200 m；云底 ~1 km，云顶 ~2 km） | `mountain.sh` / `desert.sh` / `city.sh` / `DZVAR_ortho` |
| `clouds/L12km.1.BOMEX.005.htcp` | 海洋信风积云 | 12.8 km × 12.8 km，25 m | 云层 0.5–1.5 km | `sea.sh`、通量图 |
| `clouds/L25.1.FIRE.012.htcp` | 海洋层积云 | 25 km × 25 km，100 m | 10 m 层，云 300–600 m | `L25_FIRE.sh` |

体素场：液态水混合比 `RCT`（kg/kg dry air）、温度 `T`、气压 `PABST`、水汽 `RVT`。光学系数当场算，不预存 k。

`mountain.sh` 带 `-r`：云在 XY **周期重复**。Z 不周期。`desert.sh` / `city.sh` **不** 带 `-r`，云只出现在 LES 盒子里。

### 1.4 云滴光学（htmie）

`Mie_LUT_Cloud-2-10-0.010.nc`：Mishchenko Lorenz–Mie，对数正态谱 \(r_e=10\,\mu\mathrm{m}\)、\(v_e=0.01\)。

运行时 **不用 Mie 相函数表**。只用：

- 谱段平均质量吸收/散射截面 \(\overline{C}_a(\Delta\nu)\)、\(\overline{C}_s(\Delta\nu)\)（线性滤波）
- **逐波长** 不对称因子 \(g(\lambda)\)，喂给 HG

短波体积相函数 = HG(\(g(\lambda)\))。这是求解器选择，不是 LUT 缺数据。

### 1.5 气体（htgop / ecRad k-distribution）

`ecrad_opt_prop.txt`：

- 165 个高度层 / 164 层；地面气压 101300 Pa；地面温度 294.2 K
- **16 个 LW 谱段、14 个 SW 谱段**（RRTMG 相关 k）
- 每层每谱段：若干 quadrature 点及权重；名义 \(k_a,k_s\)；云内按水汽摩尔分数 \(x_{\mathrm{H_2O}}\) 查表插值
- LW 气体 \(k_s=0\)（无 Rayleigh）；SW 气体有 \(k_a\) 与 \(k_s\)（Rayleigh）

气体是 **1D 垂直层结**，忽略地球曲率。水平均匀，除非云盒子里用 LES 的 \(x_{\mathrm{H_2O}}\) 替换名义廓线。

### 1.6 太阳（解析图元，无网格）

不是 mesh。参数：

- 半张角 \(\theta_\odot = 4.6524\times 10^{-3}\,\mathrm{rad}\)
- 立体角 \(\Omega_\odot = 2\pi(1-\cos\theta_\odot)\)
- 色温 \(T_\odot=5778\,\mathrm{K}\)（Planck）
- 主方向由方位/高度角给出（§4）

### 1.7 传感器

| 类型 | 选项 | Starter Pack 例子 |
|------|------|-------------------|
| 针孔透视 | `-C`，`lens-radius=0` | `mountain.sh` |
| 薄透镜 | `-C`，`lens-radius>0` | `city_thin_lens.sh` |
| 正交相机 | `-P` | `DZVAR_ortho`、`desert_ortho.sh` |
| 矩形通量图 | `-p`，余弦半球出射 | `L12Km_BOMEX_desc_flux_sw.sh` |

### 1.8 加速结构（不是物理图元）

- 云：**3D 八叉树**（Star-VX），叶节点存 \(k_{\min},k_{\max}\)
- 气体：**沿 Z 的二叉树**（1D SVX）
- 每 (谱段, quadrature) 一棵树；`-O cache` 可复用
- 合并判据：\((k_{\mathrm{ext,max}}-k_{\mathrm{ext,min}})\Delta z \le \tilde\tau\)，默认 \(\tilde\tau=1\)（`-T`）

---

## 2. 无偏 MC 的机理

### 2.1 积分与算法一一对应

像素值是辐亮度（或 CIE 刺激值）的积分。每条路径是该积分的一个无偏样本。期望等于被积函数，方差随 spp 降。

短波 CIE 通道 \(C\in\{X,Y,Z\}\)：

\[
C = \int_{\lambda_{\min}}^{\lambda_{\max}} L(\mathbf{x}_c,\boldsymbol{\omega}_c,\lambda)\,\bar{c}(\lambda)\,d\lambda
= \mathbb{E}\!\left[\frac{L(\lambda)}{p_C(\lambda)}\right],
\quad
p_C(\lambda)=\frac{\bar{c}(\lambda)}{\int\bar{c}}
\]

\(\bar{c}\) 是 CIE 1931 配色函数（Wyman–Sloan–Shirley 2013 解析拟合）。实现：先按 \(p_C\) 抽 \(\lambda\)，再按相关 k 的 quadrature 权重抽谱段内的 \(i_{\mathrm{quad}}\)，最后用 `atmosphere_compute_radiance_sw` 估 \(L\)。

`draw_pixel_image`：`pdf` 从 nm⁻¹ 乘 \(10^9\) 变成 m⁻¹，与 \(L\) 的每米波长密度相除，得到 W·sr⁻¹·m⁻²。

### 2.2 自由程（Woodcock / 零碰撞）

非均匀 \(k(\mathbf{x})\) 上，光学厚度 \(\tau=\int k\,ds\) 的首次碰撞服从

\[
P(\tau>t)=\mathrm{e}^{-t},\qquad \tau\sim\mathrm{Exp}(1).
\]

叶节点 majorant \(\hat k\ge k\)。在叶内：

1. 抽 \(\tau\sim\mathrm{Exp}(1)\)
2. 若 \(\tau > \hat k\cdot\Delta s_{\mathrm{leaf}}\)：叶内无碰，\(\tau\leftarrow\tau-\hat k\Delta s\)，进下一叶
3. 否则候选距离 \(\Delta s=\tau/\hat k\)，以概率 \(k/\hat k\) **接受真碰撞**，否则零碰撞并重新抽 \(\tau\)

这是最大截面法，无偏。零碰撞 ≡ 前向 δ 散射。

短波 **散射游走** 的 \(k\) 是 **\(K_s=k_{s,\mathrm{gas}}+k_{s,\mathrm{particle}}\)**，不是 \(K_{\mathrm{ext}}\)。吸收另走 analog 透过率（§2.3）。长波游走用 \(K_{\mathrm{ext}}\)，碰撞后再 Bernoulli 分成吸收/散射。

### 2.3 透过率（分解追踪）

NEE 与吸收权需要 \(\mathrm{Tr}=\exp(-\int k\,ds)\)。htrdr 用 **analog 下界 + 残差 delta-tracking**（decomposition tracking）：

- 叶上已知 \(k_{\min}\le k\le k_{\max}\)
- analog 累积 \(T_{\min}\mathrel{+}=\Delta s\cdot k_{\min}\)
- 在残差场 \(k_{\max}-k_{\min}\) 上做 Woodcock
- 若残差全程无真碰：\(\mathrm{Tr}=\mathrm{e}^{-T_{\min}}\)
- 若残差有真碰：\(\mathrm{Tr}=0\)

这是 \(\mathrm{e}^{-\tau}\) 的无偏（高方差）估计：要么 \(\mathrm{e}^{-\tau_{\min}}\)，要么 0。比逐体素累加 \(\mathrm{e}^{-\Delta\tau}\) 更贴八叉树。

短波里该估计出现两次：

- `HTSKY_Kext`：太阳可见性
- `HTSKY_Ka`：两碰撞点之间的吸收权 \(\xi\)

### 2.4 竞争风险（体积里气 vs 粒）

合并 \(K_s\) 游走到碰撞点后 **只掷一次**：

\[
P(\text{Rayleigh})=\frac{k_{s,\mathrm{gas}}}{k_{s,\mathrm{gas}}+k_{s,\mathrm{particle}}},\qquad
P(\text{HG})=1-P(\text{Rayleigh}).
\]

这与「两条独立时钟再比谁先到」等价，不要再乘一次 \(k_s/k_{\mathrm{tot}}\)。

### 2.5 局部估计（NEE）+ 路径权

短波权重（无表面俄罗斯轮盘时）：

\[
w = \underbrace{\mathbf{1}_{\boldsymbol{\omega}_c\in\Omega_\odot}\,L_\odot\,\mathrm{Tr}_{\mathrm{ext}}(\mathbf{x}_c,\boldsymbol{\omega}_c)}_{\text{直射，仅相机射线}}
+ \sum_{\text{碰撞 }i}
\xi_i\,L_\odot\,\mathrm{Tr}_{\mathrm{ext}}(\mathbf{x}_i,\boldsymbol{\omega}_\odot)\,\frac{R_i}{p_\odot}
\]

\[
\xi_{i+1}=\xi_i\cdot\mathrm{Tr}_a(\mathbf{x}_i\to\mathbf{x}_{i+1}),\quad \xi_0=1
\]

- 体积：\(R=p(\boldsymbol{\omega}_o,\boldsymbol{\omega}_\odot)\)（相函数，已含 \(1/(4\pi)\)）
- Lambert：\(R=f_r(\boldsymbol{\omega}_o,\boldsymbol{\omega}_\odot)\,(\mathbf{n}\cdot\boldsymbol{\omega}_\odot)=(\rho/\pi)(\mathbf{n}\cdot\boldsymbol{\omega}_\odot)\)
- 镜面：仅当反射方向落入太阳锥；\(R=\rho_{\mathrm{spec}}\)，\(p_\odot=1\)（Dirac）

\(p_\odot=1/\Omega_\odot\)。地面若挡住太阳，\(\mathrm{Tr}=0\)。

表面俄罗斯轮盘：以 \(P=\rho_{\mathrm{bounce}}\) 继续（Lambert 下 `ssf_bsdf_sample` 返回反照率 \(\rho\)）。失败则终止。继续时不把 \(\rho\) 乘进 \(\xi\)——与「权乘 \(\rho\) 再除 \(P=\rho\)」抵消，Lambert 无偏。

### 2.6 长波（发射追踪）

无太阳。用 \(K_{\mathrm{ext}}\) Woodcock：

- 吸收：\(w=B_\lambda(T(\mathbf{x}))\)，停
- 散射：HG 改方向，继续（长波气体 \(k_s=0\)，散射几乎只有云）
- 打地：按 BSDF 采样；俄罗斯轮盘失败则 \(w=B_\lambda(T_{\mathrm{mtl}})\)
- 穿出大气上界：\(w=0\)（太空无入射）

这是吸收处的 Planck 局部估计，无偏。

### 2.7 无偏不保证低方差

八叉树 \(\tilde\tau\)、spp、CIE 重要性采样只改方差。改 HG 为 Mie、改曝光、给 miss 垫色，都会引入偏差。

---

## 3. 传感器采样

### 3.1 透视针孔（`mountain.sh`）

像素 \((i,j)\)、样本 \(s\)：

\[
u=\frac{i+\xi_1}{W},\quad v=\frac{j+\xi_2}{H},\quad \xi\sim U[0,1)
\]

`scam_generate_ray` 从胶片点过针孔得 \((\mathbf{o},\boldsymbol{\omega})\)。`lens-radius=0` 时透镜样本被忽略。

`mountain.sh`：

- `pos = (10245.37, 615.02, 422.34)` m
- `tgt = (-2977.39, 6829.40, 842.40)` m
- `up = (0,0,1)`，`fov = 30°`（垂直）
- `def = 1280×720`，`spp = 256`

CIE 模式每个像素 **3×256** 条路径（X/Y/Z 各 256）。

### 3.2 正交相机

平行投影。胶片高度 `-P height=`，宽度由宽高比保证方像素。

### 3.3 通量图

像素内均匀抽位置。若向上打地且命中材料不是 `sky_mtl`（默认 `air`），样本拒绝。方向：相对传感器法线的 **余弦半球**。通量密度 W·m⁻²。

### 3.4 光谱默认（图像）

不写 `-s` 即为 `cie_xyz`，\(\lambda\in[380,780]\) nm。每个通道：

1. 用该通道 CDF 抽谱带，带内再解配色函数的二次近似，得到 \(\lambda\)
2. `htsky_find_spectral_band(λ)` → RRTMG 谱段
3. `htsky_spectral_band_sample_quadrature(u, iband)` → k 点

显式 `-s sw=λmin,λmax` 或 `lw=...` 则按 Planck(\(T_{\mathrm{ref}}\)) 抽波长：短波默认 \(T_{\mathrm{ref}}=5778\,\mathrm{K}\)，长波 290 K。

---

## 4. 太阳逻辑

### 4.1 方向

`-D az,el`，度。右手系，方位 0° = +X，逆时针；高度 0° = XY 平面，90° = +Z。

\[
\boldsymbol{\omega}_\odot =
\bigl(\cos el\cos az,\;
\cos el\sin az,\;
\sin el\bigr)
\]

`mountain.sh`：`225,30` → 西南、仰角 30°。

### 4.2 盘与立体角

\[
\theta_\odot=4.6524\times 10^{-3},\quad
\Omega_\odot=2\pi(1-\cos\theta_\odot),\quad
p_\odot=\frac{1}{\Omega_\odot}\approx 1.4715\times 10^{4}\,\mathrm{sr}^{-1}
\]

NEE 方向：`ssp_ran_sphere_cap_uniform_local` 在 \(\cos\theta\ge\cos\theta_\odot\) 的球冠上均匀采样，再乘太阳坐标系。

`htrdr_atmosphere_sun_is_dir_in_solar_cone`：\(\boldsymbol{\omega}\cdot\boldsymbol{\omega}_\odot \ge \cos\theta_\odot\)。

### 4.3 辐亮度

\[
L_\odot(\lambda)=B_\lambda(T_\odot),\quad T_\odot=5778\,\mathrm{K}
\]

Planck（SI）：

\[
B_\lambda(T)=\frac{2hc^{2}}{\lambda^{5}}\frac{1}{\mathrm{e}^{hc/(\lambda kT)}-1}
\quad[\mathrm{W}\cdot\mathrm{m}^{-2}\cdot\mathrm{sr}^{-1}\cdot\mathrm{m}^{-1}]
\]

\(h=6.62607015\times 10^{-34}\)，\(c=299792458\)，\(k=1.380649\times 10^{-23}\)。波长从 nm 转 m。

**太阳不是谱段平均辐亮度**，是当前样本波长上的单色 Planck。谱段只用于气体/云的 k。

### 4.4 直射 vs 漫射

`ATMOSPHERE_RADIANCE_DIRECT`：仅当 **当前射线方向已在太阳锥内** 时，加 \(L_\odot\mathrm{Tr}_{\mathrm{ext}}\)。相机射线对着太阳盘才会有日盘。漫射循环里 **不会** 在 miss 上再加日盘。

`ATMOSPHERE_RADIANCE_DIFFUSE`：碰撞点 NEE。图像模式两者都开。

---

## 5. 「天空」与环境光

没有环境光贴图。短波环境 = 太阳 + 气体/云散射。

### 5.1 气体采样（1D 层结）

高度 \(z\) → `htgop position_to_layer_id`。当前 `(iband, iquad)`：

- **云外**：该层名义 \(k_a^{\mathrm{nom}}, k_s^{\mathrm{nom}}\)
- **云内**：\(x_{\mathrm{H_2O}}=r_{vt}/(r_{vt}+M_{\mathrm{H_2O}}/M_{\mathrm{air}})\)，\(M_{\mathrm{air}}=0.0289644\)，\(M_{\mathrm{H_2O}}=0.01801528\)；再查层表

\[
k_{s,\mathrm{gas}}^{\mathrm{SW}}=k_s(z,\nu_i,q),\qquad
k_{a,\mathrm{gas}}^{\mathrm{SW}}=k_a(z,\nu_i,q)
\]

LW：\(k_{s,\mathrm{gas}}=0\)。

气体树沿 Z 建，Woodcock 的 \(\hat k\) 来自层内 min/max。

相函数：Rayleigh

\[
p_{\mathrm{R}}(\mu)=\frac{3}{16\pi}(1+\mu^{2}),\quad \mu=\boldsymbol{\omega}_{\mathrm{in}}\cdot\boldsymbol{\omega}_{\mathrm{out}}
\]

采样：\(\phi\sim U[0,2\pi)\)，\(\mu=s-1/s\)，\(s=\sqrt[3]{u+\sqrt{1+u^{2}}}\)，\(u=4\xi-2\)（与 \(1+\mu^{2}\) 的 CDF 一致）。SSF 约定 \(\boldsymbol{\omega}_o\) 指离散射点，求 \(\mu\) 前要取反。

### 5.2 天空 miss

短波：气体树与地面都 miss → **循环结束，不再加源**。权重里只剩已经累加的直射/NEE。太空 = 黑。

长波 miss = 0。

「天空颜色」= 相机射线在气体里 Rayleigh（及云 HG）散射后，对太阳 NEE 的贡献。

### 5.3 体积射线如何切气体/云

`htsky_trace_ray`（给定 `iband, iquad`）：

1. 若射线与云 AABB 无交（周期时只裁 Z）：只走气体二叉树
2. 否则：气体（到云盒）→ 云八叉树 → 气体（出云盒之后）
3. `-r` 时云盒 XY 用与地面相同的 slab，最多 32 格

filter 在叶上做 Woodcock；返回 0 表示真碰撞，停止遍历。

---

## 6. 散射介质（云）采样

### 6.1 粒子消光

体素 \((i,j,k)\)：

\[
\rho_{\mathrm{da}}=\frac{P\,M_{\mathrm{air}}}{T\,R},\quad
R=8.3144598,\;
q_l=\rho_{\mathrm{da}}\cdot\mathrm{RCT}
\]

\[
k_{a,\mathrm{p}}=\overline{C}_a(\Delta\nu)\,q_l,\qquad
k_{s,\mathrm{p}}=\overline{C}_s(\Delta\nu)\,q_l
\]

\(\overline{C}_{a,s}\) 是 **整个 RRTMG 谱段** 上 Mie 截面的线性平均，**不是** 当前 \(\lambda\) 的截面。同一谱段内所有 CIE 样本共用这对截面。\(g\) 则是 \(g(\lambda)\)，不是谱段平均 \(g_{\mathrm{avg}}\)（`g_avg` 算了但短波 setup 不用）。

不规则 Z：`svx2htcp_z` LUT 把均匀查询映射到 HTCP 层。

周期：`world_to_cloud` 把世界 XY 折进 LES 盒（分数部，负值 +1）。Z 不折。

### 6.2 HG

\[
p_{\mathrm{HG}}(\mu)=\frac{1}{4\pi}\frac{1-g^{2}}{(1+g^{2}-2g\mu)^{3/2}},\quad g=g(\lambda)\in(-1,1)
\]

采样：`ssp_ran_sphere_hg`（Henyey–Greenstein 逆 CDF）。`ssf_phase_eval` 与 pdf 相同（相函数已归一化到 \(4\pi\)）。

### 6.3 八叉树 majorant

叶存粒子 \(k_{\min},k_{\max}\)（该叶覆盖的 HTCP 细胞）。Woodcock 用 `HTSKY_SVX_MAX`。合并：

\[
(k_{\mathrm{ext,max}}-k_{\mathrm{ext,min}})\Delta z \le \tilde\tau
\]

`-T 1` 即 Villefranque 等建议的量级。`-V x,y,z` 可限制最细网格；有 cache 时 `-T/-V` 忽略。

---

## 7. 地面采样逻辑

### 7.1 求交

每步先 `htrdr_atmosphere_ground_trace_ray`，再把体积追踪的 `range[1]` 截到 `s3d_hit.distance`。体积碰撞距离若小于地面，则地面不算。

自交：命中表面后保存 `s3d_hit_prev`，下次从该图元出发。体积散射则清空。

周期地面：射线在 XY 上走周期格（`htrdr_slab_trace_ray`，最多 32 步），每格把原点变到局部 OBJ 再 Embree。Z 用几何 AABB。

### 7.2 命中哪一侧材料

`usemtl air:sand`：法线与入射方向判断 front/back，取 **对面** 材料（光线穿入的那一侧）。从空气打山，用 `sand`。`air` 为 `none`，数值误差拾到空材料时回退另一侧。

### 7.3 Lambert

\[
f_r=\frac{\rho(\lambda)}{\pi},\qquad
p(\boldsymbol{\omega}_i)=\frac{(\mathbf{n}\cdot\boldsymbol{\omega}_i)}{\pi}\ \text{（上半球）}
\]

`ssf_bsdf_sample` 返回 \(\rho\)（= \(f_r(\mathbf{n}\cdot\boldsymbol{\omega})/p\)）。NEE 用 `eval * (n·sun)`。`n·sun<0`（太阳在面下）→ \(R=0\)。

`sand` 在 380–780 nm 由 300 nm (0.05) 与 600 nm (0.16) 节点决定（MruMtl 在节点间抽一条 BRDF）。

### 7.4 镜面（海面等）

反射方向 \(\boldsymbol{\omega}_r=2(\mathbf{n}\cdot\boldsymbol{\omega}_o)\mathbf{n}-\boldsymbol{\omega}_o\)。仅当 \(\boldsymbol{\omega}_r\) 在太阳锥内才有直接贡献；pdf=1。透射分量丢弃（`SSF_REFLECTION` 才保留）。

### 7.5 地面温度

只进长波发射。短波忽略 `mountain.mtls` 里的 300 K。

---

## 8. 短波仿真管线（逐步）

对应 `atmosphere_compute_radiance_sw`。

```
输入: 位置 pos, 方向 dir, 波长 λ, 谱段 iband, 四边形点 iquad
L_sun = Planck(λ, 5778 K)
g     = htmie_fetch_g(λ)
ξ = 1,  w = 0

若 dir ∈ 太阳锥:
    若地面挡住: Tr=0
    否则 Tr = transmissivity(Kext)     # 分解追踪
    w += L_sun * Tr                    # 直射日盘

循环:
    地面求交 → t_surf
    τ ~ Exp(1)
    体积追踪 range=[0, t_surf]，Woodcock(Ks) → 碰撞距离 t
    若地面与体积都 miss: break          # 太空，不再加源

    pos_next = pos + t * dir
    Tr_abs = transmissivity(Ka, 段 [0,t])
    若 Tr_abs=0: break
    ξ *= Tr_abs

    若打地:
        建 BSDF；采样 dir_next, bounce_ρ
        若镜面且 dir_next∈锥: R=bounce_ρ, pdf_sun=1
        否则: 抽太阳方向; R = f_r * (n·sun)
    否则:  # 体积
        以 ks_gas/ks 选 Rayleigh 或 HG
        采样 dir_next
        抽太阳方向; R = phase(wo, sun)

    若 R>0 且地面不挡太阳:
        Tr = transmissivity(Kext, pos_next → sun)
    否则 Tr=0

    w += ξ * L_sun * Tr * R / pdf_sun

    若打地且 u ≥ bounce_ρ: break        # 俄罗斯轮盘吸收
    pos, dir ← pos_next, dir_next

返回 w
```

核心算法：

| 步骤 | 算法 | 模块 |
|------|------|------|
| 像素/透镜 | 均匀胶片 + 针孔 | `scam` |
| λ | CIE \(\bar{x},\bar{y},\bar{z}\) 重要性 | `htrdr_ran_wlen_cie_xyz` |
| k 点 | 相关 k 权重 | `htgop` / `htsky_spectral_band_sample_quadrature` |
| 体积自由程 | Woodcock on \(K_s\) | `scattering_hit_filter` |
| 吸收权 / 太阳可见 | 分解追踪 | `transmissivity_hit_filter` |
| 气/粒 | 一次 Bernoulli | `htsky_fetch_raw_property` |
| 相函数 | HG / Rayleigh | `ssf_phase_*` |
| 地面 | Embree + 周期 slab | `s3d` / `htrdr_slab` |
| 太阳 NEE | 球冠均匀 | `htrdr_atmosphere_sun` |

---

## 9. 长波仿真管线

`atmosphere_compute_radiance_lw`（Starter Pack 的 `lw_ortho_*` 走这条，不是 `mountain.sh`）。

```
循环:
    地面求交
    Woodcock(Kext) 与地面比远近
    miss 两者: w=0; break              # 太空
    若吸收: w = Planck(λ, T(x)); break
    若散射: HG 采样新方向
    若打地:
        采样 BSDF
        俄罗斯轮盘失败: w = Planck(λ, T_mtl)
        成功: 继续反射
返回 w
```

温度：云内用 HTCP `T`；云外用 htgop 层间线性插值。

---

## 10. 按图元：管线 / 采样 / 数学底座

### 10.1 太阳盘

| | |
|--|--|
| 几何 | 方向球冠，无网格 |
| 采样 | 直射：视方向 ∈ 锥则贡献；NEE：球冠均匀 |
| 底座 | \(\Omega=2\pi(1-\cos\theta_\odot)\)；\(L=B_\lambda(5778\,\mathrm{K})\) |
| 遮挡 | 地面射线；体积 `Tr(Kext)` |

### 10.2 气体

| | |
|--|--|
| 几何 | 1D 层，Z 二叉树 |
| 自由程 | 与粒子合并的 \(K_s\) Woodcock |
| 相函数 | Rayleigh \(3(1+\mu^2)/(16\pi)\) |
| k | ecRad 相关 k；云内 \(x_{\mathrm{H_2O}}\) 插值 |
| 发射 | 仅 LW：吸收处 Planck |

### 10.3 云滴

| | |
|--|--|
| 几何 | LES 直角网格；查询可 XY 周期 |
| 自由程 | 同上，\(k_{s,\mathrm{p}}=\overline{C}_s q_l\) |
| 相函数 | HG(\(g(\lambda)\))，**不是** Mie LUT |
| 加速 | 八叉树 min/max，\(\tilde\tau=1\) |
| 发射 | 仅 LW |

### 10.4 地面三角

| | |
|--|--|
| 几何 | OBJ + Embree；可选 XY 周期 |
| 采样 | BSDF 采样改方向；NEE 对太阳求 `eval` |
| Lambert | \(f_r=\rho/\pi\)，余弦半球 |
| 镜面 | 反射 Dirac，对太阳锥检验 |
| 终止 | 俄罗斯轮盘 \(P=\rho\) |
| 发射 | 仅 LW，\(T_{\mathrm{mtl}}\) |

### 10.5 传感器

| | |
|--|--|
| 针孔 | 胶片均匀 → 透视射线 |
| 薄透镜 | 额外透镜圆盘，焦平面 |
| 正交 | 平行射线 |
| 通量 | 面均匀 × 余弦半球；拒绝埋入固体的样本 |

### 10.6 「天空」

不是图元。气体层 + miss=0 + 太阳。材料名 `air` 只用于通量图判断「外侧是大气」。

---

## 11. 核心模块

```
htrdr-atmosphere
 ├─ atmosphere_draw_map          像素循环、CIE/Planck 波长、spp
 ├─ atmosphere_compute_radiance_sw / _lw
 ├─ htrdr_atmosphere_sun         锥、立体角、Planck、球冠采样
 ├─ htrdr_atmosphere_ground      OBJ、周期 slab、接口材料
 ├─ htrdr_materials / mrumtl     光谱 BRDF
 └─ htsky
      ├─ htgop   气体 k-distribution
      ├─ htcp    LES 云场
      ├─ htmie   截面与 g(λ)
      └─ svx     云八叉树 + 气体 Z 树

Star-Engine:  ssp (RNG, Exp, HG, 球冠, 余弦半球)
              s3d (Embree)
              ssf (BSDF / 相函数)
              scam (相机)
```

---

## 12. `mountain.sh` 全参数

| 参数 | 值 | 作用 |
|------|-----|------|
| `-a` | `ecrad_opt_prop.txt` | 气体 164 层，16 LW + 14 SW 谱段 |
| `-c` | `DZVAR.1.ARMCU.008.diaKCL.htcp` | 256×256×160 积云 |
| `-r` | 开 | 云 XY 周期 |
| `-m` | `Mie_LUT_Cloud-2-10-0.010.nc` | \(r_e=10\,\mu\mathrm{m}\)，\(v_e=0.01\) |
| `-g` | `mountain.obj` | 8.4M 三角，10 km × 10 km × 700 m |
| `-R` | 开 | 地面 XY 周期 |
| `-M` | `mountain.mtls` | `air` none；`sand` Lambert，300 K |
| `-D` | `225,30` | 太阳方位 225°、高度 30° |
| `-C` | pos/tgt/up/fov=30 | 针孔 |
| `-i` | `def=1280x720:spp=256` | CIE 每通道 256 spp |
| `-T` | `1` | 八叉树光学厚度阈 |
| `-O` | `DZVAR_T1.cache` | 加速结构缓存 |
| `-o` | `mountain_1280x720x256.txt` | `htrdr-image` ASCII |
| `-s` | （默认） | `cie_xyz`，380–780 nm |
| `-n` | （默认 `air`） | 天空材料名 |
| `-V` | （默认 UINT_MAX） | 树分辨率 = 云网格 |
| 太阳半角 | 硬编码 `4.6524e-3` rad | 不在命令行 |
| \(T_\odot\) | 硬编码 5778 K | |
| 周期 slab 上限 | 硬编码 32 | 地面与云 |

显示：`htpp` 把 `htrdr-image` 转 PPM（filmic / exposure）。那是色调映射，不是求解器的一部分。

---

## 13. Starter Pack 其它脚本（同一求解器）

| 脚本 | 云 | 地面 | 太阳 | 传感器 | 与 mountain 的差别 |
|------|----|------|------|--------|-------------------|
| `desert.sh` | DZVAR，**无 `-r`** | desert `-R` | 130°, 5° | fov 70° | 云不周期；低太阳 |
| `city.sh` | DZVAR，无 `-r` | city `-R` | 100°, 40° | fov 70° | 多材料城市 |
| `sea.sh` | BOMEX **`-r`** | sea `-R` | 0°, 50° | fov 70° | 镜面水 |
| `desert_ortho.sh` / `DZVAR_ortho` | DZVAR | 沙漠或无 | 正交向下 | TOA 正交图 | |
| `L12Km_BOMEX_desc_flux_sw.sh` | BOMEX `-r` | 无地面 | 0°, 90° | 12.8 km 通量图 | `-s sw=380,400` |
| `L25_FIRE.sh` | FIRE | — | — | 透视 | 层积云 |
| `city_thin_lens.sh` | 同 city | city | 同 city | `lens-radius>0` | 景深 |

长波正交图在 pack 里是预渲染产物；算法见 §9。

---

## 14. 源码锚点

| 主题 | 路径 |
|------|------|
| 短波路径 | `htrdr/src/atmosphere/htrdr_atmosphere_compute_radiance_sw.c` |
| 长波路径 | `.../htrdr_atmosphere_compute_radiance_lw.c` |
| 像素 / CIE | `.../htrdr_atmosphere_draw_map.c` |
| 太阳 | `.../htrdr_atmosphere_sun.c` |
| 地面 | `.../htrdr_atmosphere_ground.c` |
| 周期求交 | `htrdr/src/core/htrdr_slab.c` |
| Planck / CIE 范围 | `htrdr/src/core/htrdr_spectral.h`, `htrdr_ran_wlen_cie_xyz.c` |
| 体积 k / 周期 | `local/src/htsky/src/htsky.c` |
| 气体树 | `.../htsky_atmosphere.c` |
| 云树 | `.../htsky_cloud.c` |
| 气+云射线 | `.../htsky_svx.c` `htsky_trace_ray` |
| HG / Rayleigh / Lambert | `star-sf`：`ssf_phase_hg.c`, `ssf_phase_rayleigh.c`, `ssf_lambertian_reflection.c` |
| 场景脚本 | `htrdr-atmosphere-spk/htrdr-Atmosphere-Starter-Pack-0.8.0/mountain.sh` |

---

## 15. 和「看起来像环境光」的对应关系

| 观感 | htrdr 实际 |
|------|------------|
| 蓝天 | 气体 Rayleigh + 太阳 NEE，不是 miss 色 |
| 太阳圆盘 | 相机射线落在 \(\Omega_\odot\) 内时的直射项 |
| 云亮核 | HG 前向峰 × 太阳 NEE × \(\mathrm{Tr}(K_{\mathrm{ext}})\) × 多次散射 |
| 地面沙色 | Lambert \(\rho(\lambda)\) × 太阳 NEE × 天空散射入射 |
| 云影 | 地上 NEE 的 \(\mathrm{Tr}(K_{\mathrm{ext}})\) 被云挡住 |
| 地平线暗 | 路径更长，吸收权 \(\mathrm{Tr}_a\) 与几何 |
| 太空黑 | SW/LW miss 贡献为 0 |

# 融合前端机器视觉与机理耦合时序网络的路段动态风险实时预判研究

本文构建一种融合前端机器视觉测量、交通冲突机理计算与时序神经网络预测的路段动态风险实时预判方法。整体思路是：首先从监控视频中实时检测与跟踪车辆，结合多区域透视标定将图像坐标转换为道路平面米制坐标；其次根据车辆二维运动状态计算可解释的瞬时交通冲突风险，得到每个研究区域的区域瞬时事故概率；最后将连续时刻的机理风险序列与交通状态辅助特征输入时序网络，学习风险由短时扰动向持续高危状态演化的规律，实现未来时段路段事故风险预判。

## 一、前端机器视觉测量层

设监控视频第 $t$ 帧图像为 $I(t)$，利用 YOLO 车辆检测模型提取图像中的车辆目标：

$$
\mathcal{D}(t)=f_{\mathrm{YOLO}}\left(I(t);\Theta_y\right)
$$

其中，$f_{\mathrm{YOLO}}(\cdot)$ 表示参数为 $\Theta_y$ 的 YOLO 检测模型。检测结果集合为：

$$
\mathcal{D}(t)=\left\{d_1(t),d_2(t),\ldots,d_{M(t)}(t)\right\}
$$

单个检测目标定义为：

$$
d_i(t)=
\left[
b_{i1}(t),b_{i2}(t),b_{i3}(t),b_{i4}(t),
p_i^{\mathrm{det}}(t),c_i
\right]
$$

式中，$b_{i1}(t),b_{i2}(t)$ 为检测框左上角像素坐标，$b_{i3}(t),b_{i4}(t)$ 为检测框右下角像素坐标，$p_i^{\mathrm{det}}(t)$ 为检测置信度，$c_i$ 为车型类别编码。实际计算中保留 car、motorcycle、bus、truck 等车辆类别。

考虑到检测框底边容易受到车辆遮挡、阴影和框体抖动影响，本文采用检测框中心点作为车辆图像测量锚点：

$$
\mathbf{u}_i(t)=
\begin{bmatrix}
u_i(t)\\
v_i(t)
\end{bmatrix}
=
\begin{bmatrix}
\dfrac{b_{i1}(t)+b_{i3}(t)}{2}\\[6pt]
\dfrac{b_{i2}(t)+b_{i4}(t)}{2}
\end{bmatrix}
$$

随后使用多目标跟踪算法 ByteTrack 对跨帧车辆身份进行关联，得到稳定车辆轨迹编号：

$$
\tau_i=\left\{\mathbf{u}_i(t_0),\mathbf{u}_i(t_0+1),\ldots,\mathbf{u}_i(t)\right\}
$$

式中，$\tau_i$ 表示车辆 $i$ 在视频中的像素轨迹。

## 二、多区域透视标定与道路平面坐标恢复

实际道路监控可能包含多个感兴趣区域，例如不同方向车道、匝道汇入区或路口不同冲突区。设第 $r$ 个研究区域在图像平面中的多边形为 $A_r^{\mathrm{img}}$，对应真实道路平面区域为 $A_r^{\mathrm{road}}$。车辆 $i$ 属于区域 $r$ 的判定为：

$$
\mathbf{u}_i(t)\in A_r^{\mathrm{img}}
$$

若车辆锚点不属于任何研究区域，则该车辆不参与后续区域风险计算；若属于某一区域，则使用该区域独立标定参数进行测量。

第 $r$ 个区域由四组图像标定点与真实道路平面点建立对应关系：

$$
\mathbf{u}_{rk}=
\begin{bmatrix}
u_{rk}\\
v_{rk}\\
1
\end{bmatrix},
\quad
\mathbf{x}_{rk}=
\begin{bmatrix}
x_{rk}\\
y_{rk}\\
1
\end{bmatrix},
\quad k=1,2,3,4
$$

通过四点透视标定求解区域单应矩阵 $\mathbf{H}_r$：

$$
\lambda_k\mathbf{x}_{rk}=\mathbf{H}_r\mathbf{u}_{rk},
\quad k=1,2,3,4
$$

对任意车辆图像锚点 $\mathbf{u}_i(t)$，其道路平面齐次坐标为：

$$
\begin{bmatrix}
\tilde{x}_i(t)\\
\tilde{y}_i(t)\\
\tilde{w}_i(t)
\end{bmatrix}
=
\mathbf{H}_r
\begin{bmatrix}
u_i(t)\\
v_i(t)\\
1
\end{bmatrix}
$$

归一化后得到米制道路平面坐标：

$$
x_i(t)=\frac{\tilde{x}_i(t)}{\tilde{w}_i(t)},
\quad
y_i(t)=\frac{\tilde{y}_i(t)}{\tilde{w}_i(t)}
$$

因此，在标定正确、参考距离测量准确的前提下，该方法可适配鸟瞰视角、街角监控视角和斜向道路视角。不同区域分别使用各自的 $\mathbf{H}_r$，避免将不同路面平面或不同方向车道强行放入同一坐标映射。

## 三、车辆二维运动状态向量

设视频帧率为 $F$，同一车辆在时间窗口 $[t-\Delta t,t]$ 内的道路平面坐标分别为 $\left(x_i(t-\Delta t),y_i(t-\Delta t)\right)$ 与 $\left(x_i(t),y_i(t)\right)$。车辆二维速度分量为：

$$
v_{ix}(t)=\frac{x_i(t)-x_i(t-\Delta t)}{\Delta t}
$$

$$
v_{iy}(t)=\frac{y_i(t)-y_i(t-\Delta t)}{\Delta t}
$$

为削弱目标检测抖动和跟踪微小漂移的影响，对速度序列进行短窗口平滑，记为 $\bar{v}_{ix}(t)$ 与 $\bar{v}_{iy}(t)$。加速度分量定义为：

$$
a_{ix}(t)=
\frac{\bar{v}_{ix}(t)-\bar{v}_{ix}(t-\delta t)}{\delta t}
$$

$$
a_{iy}(t)=
\frac{\bar{v}_{iy}(t)-\bar{v}_{iy}(t-\delta t)}{\delta t}
$$

车辆航向角由平滑速度方向给出：

$$
\theta_i(t)=
\operatorname{atan2}
\left(
\bar{v}_{iy}(t),
\bar{v}_{ix}(t)
\right)
$$

由此得到单车辆二维运动状态向量：

$$
o_i(t)=
\left[
x_i(t),y_i(t),
v_{ix}(t),v_{iy}(t),
a_{ix}(t),a_{iy}(t),
\theta_i(t),c_i
\right]
$$

其中，$x_i(t),y_i(t)$ 的单位为 m，$v_{ix}(t),v_{iy}(t)$ 的单位为 m/s，$a_{ix}(t),a_{iy}(t)$ 的单位为 m/s^2，$\theta_i(t)$ 的单位为 rad。

第 $r$ 个研究区域内的车辆状态集合为：

$$
O_r(t)=
\left\{
o_i(t)\mid \mathbf{u}_i(t)\in A_r^{\mathrm{img}}
\right\}
$$

区域车辆数为 $N_r(t)=|O_r(t)|$。区域面积由真实道路平面标定四边形计算得到，记为 $S_{A_r}$，则区域车流密度为：

$$
\rho_r(t)=\frac{N_r(t)}{S_{A_r}}
$$

## 四、机理瞬时风险量化层

机理层的目标是将单帧车辆状态集合 $O_r(t)$ 转换为可解释的区域瞬时事故概率 $P_{A_r}(t)$。该层不直接学习黑箱风险，而是基于车辆间相对位置、速度方向和碰撞剩余时间进行计算。

为适配不同拍摄角度和道路朝向，在道路平面内以车辆速度方向建立局部纵向轴：

$$
\mathbf{e}_{i}^{\mathrm{long}}(t)=
\frac{
\begin{bmatrix}
v_{ix}(t)\\
v_{iy}(t)
\end{bmatrix}
}{
\sqrt{v_{ix}^{2}(t)+v_{iy}^{2}(t)}
}
$$

对应局部横向轴为：

$$
\mathbf{e}_{i}^{\mathrm{lat}}(t)=
\begin{bmatrix}
-e_{iy}^{\mathrm{long}}(t)\\
e_{ix}^{\mathrm{long}}(t)
\end{bmatrix}
$$

车辆 $i$ 到车辆 $j$ 的相对位置为：

$$
\Delta \mathbf{x}_{ij}(t)=
\begin{bmatrix}
x_j(t)-x_i(t)\\
y_j(t)-y_i(t)
\end{bmatrix}
$$

则相对纵向间距和侧向间距分别为：

$$
s_{\mathrm{long},ij}(t)=
\Delta \mathbf{x}_{ij}(t)\cdot
\mathbf{e}_{i}^{\mathrm{long}}(t)
$$

$$
s_{\mathrm{lat},ij}(t)=
\Delta \mathbf{x}_{ij}(t)\cdot
\mathbf{e}_{i}^{\mathrm{lat}}(t)
$$

车辆速率为：

$$
v_i(t)=
\sqrt{v_{ix}^{2}(t)+v_{iy}^{2}(t)}
$$

### 4.1 纵向追尾风险

当两车同向行驶、车辆 $j$ 位于车辆 $i$ 前方且车辆 $i$ 相对车辆 $j$ 存在纵向闭合速度时，定义纵向追尾冲突时间：

$$
\mathrm{TTC}_{ij}(t)=
\begin{cases}
\dfrac{s_{\mathrm{long},ij}(t)}
{
\left(\mathbf{v}_i(t)-\mathbf{v}_j(t)\right)
\cdot
\mathbf{e}_{i}^{\mathrm{long}}(t)
},
& s_{\mathrm{long},ij}(t)>0
\text{ 且 }
\left(\mathbf{v}_i(t)-\mathbf{v}_j(t)\right)
\cdot
\mathbf{e}_{i}^{\mathrm{long}}(t)>0,\\[10pt]
+\infty,
& \text{其他情况}.
\end{cases}
$$

其中，$\mathbf{v}_i(t)=[v_{ix}(t),v_{iy}(t)]^T$。为避免远距离轻微接近造成风险虚高，引入即时风险时域阈值 $T_h$，并使用归一化指数衰减函数将 TTC 映射为追尾风险概率：

$$
P_{ij}^{\mathrm{long}}(t)=
\begin{cases}
\dfrac{
\exp\left(-\dfrac{\mathrm{TTC}_{ij}(t)}{\alpha}\right)
-
\exp\left(-\dfrac{T_h}{\alpha}\right)
}{
1-\exp\left(-\dfrac{T_h}{\alpha}\right)
},
& 0<\mathrm{TTC}_{ij}(t)\leq T_h,\\[12pt]
0,
& \text{其他情况}.
\end{cases}
$$

式中，$\alpha$ 为纵向风险时间衰减尺度。该映射满足：TTC 越短，风险越高；TTC 接近 $T_h$ 时，风险趋近于 0；TTC 为 $+\infty$ 时风险为 0。

### 4.2 侧向擦碰风险

侧向风险用于描述变道、并行靠近、匝道汇入等横向逼近情形。侧向相对速度定义为：

$$
\Delta v_{\mathrm{lat},ij}(t)=
\left(\mathbf{v}_i(t)-\mathbf{v}_j(t)\right)
\cdot
\mathbf{e}_{i}^{\mathrm{lat}}(t)
$$

当两车横向距离持续缩小，且纵向距离处于可能发生擦碰的门控范围 $L_g$ 内时，定义侧向冲突时间：

$$
\mathrm{LTTC}_{ij}(t)=
\begin{cases}
-\dfrac{s_{\mathrm{lat},ij}(t)}
{\Delta v_{\mathrm{lat},ij}(t)},
& s_{\mathrm{lat},ij}(t)\Delta v_{\mathrm{lat},ij}(t)<0
\text{ 且 }
\left|s_{\mathrm{long},ij}(t)\right|\leq L_g,\\[10pt]
+\infty,
& \text{其他情况}.
\end{cases}
$$

进一步映射为侧向擦碰风险概率：

$$
P_{ij}^{\mathrm{lat}}(t)=
\begin{cases}
\dfrac{
\exp\left(-\dfrac{\mathrm{LTTC}_{ij}(t)}{\beta}\right)
-
\exp\left(-\dfrac{T_h}{\beta}\right)
}{
1-\exp\left(-\dfrac{T_h}{\beta}\right)
},
& 0<\mathrm{LTTC}_{ij}(t)\leq T_h,\\[12pt]
0,
& \text{其他情况}.
\end{cases}
$$

式中，$\beta$ 为侧向风险时间衰减尺度。纵向门控 $L_g$ 的作用是排除相距较远车辆之间由检测抖动或轻微横向速度造成的虚假侧向风险。

### 4.3 区域瞬时事故概率

车辆对综合碰撞概率定义为：

$$
P_{ij}(t)=
\max
\left(
P_{ij}^{\mathrm{long}}(t),
P_{ij}^{\mathrm{lat}}(t)
\right)
$$

单车 $i$ 在当前区域内面临的最大冲突风险为：

$$
P_i(t)=
\max_{\substack{o_j(t)\in O_r(t)\\j\neq i}}
P_{ij}(t)
$$

基于单车安全概率的乘积融合，得到第 $r$ 个研究区域在第 $t$ 帧的瞬时事故概率：

$$
P_{A_r}(t)=
\begin{cases}
1-\displaystyle\prod_{o_i(t)\in O_r(t)}
\left(1-P_i(t)\right),
& N_r(t)\geq 2,\\[10pt]
0,
& N_r(t)<2.
\end{cases}
$$

机理层同时输出以下可解释风险特征：

$$
\mathbf{m}_r(t)=
\left[
P_{A_r}(t),
\rho_r(t),
N_r(t),
P_{\max,i}(t),
P_{\max,ij}(t),
\eta_r^{\mathrm{long}}(t),
\eta_r^{\mathrm{lat}}(t)
\right]
$$

其中，$P_{\max,i}(t)$ 为区域内最大单车风险，$P_{\max,ij}(t)$ 为最大车辆对风险，$\eta_r^{\mathrm{long}}(t)$ 与 $\eta_r^{\mathrm{lat}}(t)$ 表示当前区域主导风险类型，可由车辆对风险明细统计得到。

## 五、机理耦合时序网络预测层

机理模型给出的是当前帧或当前时刻的瞬时风险，能够解释“此刻为什么危险”；但事故预判还需要判断风险是否具有持续性、累积性和上升趋势。因此，时序模型不应只依赖单帧 $P_{A_r}(t)$，而应将机理概率作为主通道，并融合区域交通状态与冲突结构特征。

定义第 $r$ 个区域在第 $t$ 帧输入时序网络的特征向量：

$$
\mathbf{z}_r(t)=
\left[
P_{A_r}(t),
\rho_r(t),
N_r(t),
N_r^{\mathrm{valid}}(t),
P_{\max,i}(t),
P_{\max,ij}(t),
\eta_r^{\mathrm{long}}(t),
\eta_r^{\mathrm{lat}}(t)
\right]
$$

式中，$N_r^{\mathrm{valid}}(t)$ 表示具备有效速度和航向估计、可参与风险计算的车辆数。$P_{A_r}(t)$ 是机理风险主变量，其他变量用于补充风险来源、区域拥挤程度和车辆对冲突结构。

采用长度为 $T$ 的滑动窗口构造时序样本：

$$
\mathbf{S}_r(t)=
\left[
\mathbf{z}_r(t-T+1),
\mathbf{z}_r(t-T+2),
\ldots,
\mathbf{z}_r(t)
\right]
$$

其中，$\mathbf{S}_r(t)\in\mathbb{R}^{T\times F_z}$，$F_z$ 为单帧特征维度。输入模型前对连续变量进行标准化：

$$
\widetilde{\mathbf{z}}_r(t)=
\frac{\mathbf{z}_r(t)-\boldsymbol{\mu}}
{\boldsymbol{\sigma}+\epsilon}
$$

时序网络可采用 LSTM、GRU 或 TCN。以 LSTM 为例，其递推过程为：

$$
\mathbf{h}_k=
\operatorname{LSTMCell}
\left(
\widetilde{\mathbf{z}}_r(t-T+k),
\mathbf{h}_{k-1}
\right),
\quad k=1,2,\ldots,T
$$

窗口末端隐藏状态 $\mathbf{h}_T$ 聚合了近期风险演化信息。未来 $H$ 秒内的区域高风险概率预测为：

$$
\widehat{Y}_r(t)=
\sigma
\left(
\mathbf{W}_o\mathbf{h}_T+b_o
\right)
$$

式中，$\widehat{Y}_r(t)\in[0,1]$ 表示区域 $r$ 在未来预测时域 $H$ 内发生事故或进入高危状态的概率。

## 六、监督标签与训练目标

若存在真实事故或人工险情标注，可将未来 $H$ 秒内是否发生事故、急刹、强制变道、近碰或人工确认高危事件作为监督标签：

$$
Y_r(t)=
\begin{cases}
1,
& \exists \tau\in(t,t+H],\ \mathrm{event}_r(\tau)=1,\\[6pt]
0,
& \text{其他情况}.
\end{cases}
$$

若暂时缺少人工事故标签，可使用机理风险构造弱监督标签，先训练风险趋势预测基线：

$$
Y_r^{\mathrm{weak}}(t)=
\begin{cases}
1,
& \max_{\tau\in(t,t+H]}
P_{A_r}(\tau)\geq \gamma,\\[8pt]
0,
& \text{其他情况}.
\end{cases}
$$

式中，$\gamma$ 为高风险阈值。弱监督标签并不等价于真实事故标签，但可用于学习“未来风险是否上升到高危区间”的趋势预测能力。后续可用人工标注或真实事故记录替换弱标签，完成从风险趋势预测到事故概率预测的升级。

二分类训练目标采用加权交叉熵，以缓解高危样本稀少带来的类别不平衡：

$$
\mathcal{L}=
-w_1Y_r(t)\log \widehat{Y}_r(t)
-w_0\left(1-Y_r(t)\right)
\log\left(1-\widehat{Y}_r(t)\right)
$$

其中，$w_1$ 与 $w_0$ 分别为正负样本权重。

## 七、实时预判输出

系统实时运行时，每一帧按如下流程执行：

1. YOLO 检测车辆目标，得到检测框、置信度和车型编码；
2. ByteTrack 关联跨帧车辆身份；
3. 根据车辆中心点判定所属研究区域；
4. 使用对应区域的透视变换矩阵恢复道路平面坐标；
5. 计算车辆二维速度、加速度和航向角；
6. 在每个区域内计算车辆对 TTC、LTTC 和机理瞬时风险；
7. 生成区域特征向量 $\mathbf{z}_r(t)$ 并更新滑动窗口；
8. 将窗口序列 $\mathbf{S}_r(t)$ 输入时序网络，输出未来时段风险概率 $\widehat{Y}_r(t)$。

最终每个研究区域同时具有两个层次的风险输出：

$$
\left[
P_{A_r}(t),
\widehat{Y}_r(t)
\right]
$$

其中，$P_{A_r}(t)$ 表示当前瞬时机理风险，适合解释当前车辆对冲突；$\widehat{Y}_r(t)$ 表示未来时段动态风险，适合提前预警。二者结合后，系统既保留交通冲突模型的可解释性，又具备时序学习模型对风险累积和趋势变化的预测能力。

# 融合前端机器视觉与机理耦合时序网络的路段动态风险实时预判研究

本文构建一种融合前端机器视觉测量、交通冲突机理计算与时序神经网络预测的路段动态风险实时预判方法。整体思路是：首先从监控视频中实时检测与跟踪车辆，结合多区域透视标定将图像坐标转换为道路平面米制坐标；其次根据车辆二维运动状态计算可解释的瞬时交通冲突风险，得到每个研究区域的区域瞬时事故概率；最后将连续时刻的机理风险序列与交通状态辅助特征输入时序网络，学习风险由短时扰动向持续高危状态演化的规律，实现未来时段路段事故风险预判。

传统道路安全评价主要依赖历史事故记录，但事故事件具有低频性、随机性和统计滞后性，难以直接满足实时安全监测与主动干预需求。交通冲突理论将尚未发展为实际碰撞、但具有事故演化潜势的道路使用者交互作为替代安全观测，并通过碰撞时间、交互严重程度和风险暴露程度等指标开展主动安全评价。已有研究进一步证明，计算机视觉技术能够从道路监控视频中自动提取车辆轨迹，并据此完成交通冲突识别、碰撞风险计算和道路安全状态评价[6-9]。近年来，车辆轨迹、交通冲突指标和交通状态时间序列也逐渐被用于实时冲突预测与短时事故风险预测，为本文构建“前端机器视觉测量—瞬时交通冲突机理计算—时序动态风险预判”的整体技术路线提供了理论基础[10-12]。

## 一、前端机器视觉测量层

设监控视频第 $t$ 帧图像为 $I(t)$，利用 YOLO 车辆检测模型提取图像中的车辆目标：

$$
\mathcal{D}(t)=f_{\mathrm{YOLO}}\left(I(t);\Theta_y\right)
$$

其中，$f_{\mathrm{YOLO}}(\cdot)$ 表示参数为 $\Theta_y$ 的 YOLO 检测模型。

YOLO 将目标边界框定位与类别识别统一为端到端回归任务，通过一次神经网络前向传播直接获得图像中的目标位置、类别和检测置信度。相较于需要候选区域生成和多阶段处理的目标检测方法，YOLO 具有较高的推理效率，适合应用于连续交通监控视频中的实时车辆检测任务[1]。

检测结果集合为：

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

多目标跟踪的主要任务是在连续视频帧中为同一车辆保持一致的身份编号，并由离散检测结果恢复连续运动轨迹。ByteTrack 不仅关联高置信度检测框，还利用轨迹预测结果对部分低置信度检测框进行二次关联，从而恢复因遮挡、目标尺度变化或检测置信度下降而被遗漏的真实目标。该方法能够减少车辆轨迹中断和身份切换，为后续速度、加速度、航向角和车辆间冲突指标计算提供连续轨迹基础[2]。

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

平面单应变换描述两个投影平面之间的射影映射关系。在摄像机位置固定、研究道路局部近似为平面且标定点对应关系已知的条件下，可利用不少于四组非共线点求解图像平面与道路平面之间的单应矩阵。通过齐次坐标归一化，可以将车辆在图像中的像素位置转换为具有真实尺度的道路平面坐标。该类方法已被广泛用于交通监控摄像机标定、道路空间测量和基于视频的车辆速度估计[3-4]。

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

实际交通监控画面可能同时包含不同方向车道、道路坡度变化区域、匝道和交叉口冲突区。采用多个研究区域分别完成透视标定，相当于使用多个局部道路平面对复杂道路几何结构进行近似，可降低单一全局平面映射在不同道路区域中产生的空间尺度误差。这种分区域标定思想与交通监控场景中依据局部道路几何结构进行尺度恢复和运动测量的研究思路一致[4]。

## 三、车辆二维运动状态向量

设视频帧率为 $F$，同一车辆在时间窗口 $[t-\Delta t,t]$ 内的道路平面坐标分别为 $\left(x_i(t-\Delta t),y_i(t-\Delta t)\right)$ 与 $\left(x_i(t),y_i(t)\right)$。车辆二维速度分量为：

$$
v_{ix}(t)=\frac{x_i(t)-x_i(t-\Delta t)}{\Delta t}
$$

$$
v_{iy}(t)=\frac{y_i(t)-y_i(t-\Delta t)}{\Delta t}
$$

为削弱目标检测抖动和跟踪微小漂移的影响，对速度序列进行短窗口平滑，记为 $\bar{v}_{ix}(t)$ 与 $\bar{v}_{iy}(t)$。

车辆速度和加速度分别由道路平面位置序列的一阶和二阶时间差分获得，而目标检测误差、轨迹关联误差和坐标标定误差可能在差分过程中被进一步放大。因此，在计算运动导数前对车辆轨迹或速度序列进行短窗口平滑，是提高运动状态估计稳定性的重要步骤。Savitzky–Golay 滤波等局部多项式平滑方法能够在削弱高频噪声的同时保留信号的局部变化趋势，并支持连续信号及其导数的稳定估计[5]。

加速度分量定义为：

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

其中，$x_i(t),y_i(t)$ 的单位为 m，$v_{ix}(t),v_{iy}(t)$ 的单位为 m/s，$a_{ix}(t),a_{iy}(t)$ 的单位为 m/s²，$\theta_i(t)$ 的单位为 rad。

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

交通冲突分析是一种基于道路使用者微观交互过程的主动安全评价方法。其基本思想是将车辆之间在时间和空间上不断演化的交互视为可能进一步发展为事故的连续过程，并通过可观测的运动学变量衡量交互严重程度。与仅依赖历史事故统计的方法相比，交通冲突指标能够利用车辆位置、速度、运动方向和相对距离，对当前道路安全状态进行更高时间分辨率的刻画[6-9]。

本文将单帧车辆状态集合转换为车辆对风险和区域瞬时风险，本质上属于自动化交通冲突分析。已有概率化交通安全研究表明，可以根据道路使用者轨迹识别潜在碰撞位置，计算车辆之间的碰撞风险，并进一步在车辆、时间和道路空间层面对风险进行聚合，为本文构建车辆对风险和区域瞬时事故概率提供了理论依据[8]。

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

其中，$\mathbf{v}_i(t)=[v_{ix}(t),v_{iy}(t)]^T$。

碰撞时间（Time-to-Collision，TTC）是交通冲突研究中应用最广泛的替代安全指标之一。其基本含义是在车辆保持当前相对运动状态的条件下，从当前时刻到潜在碰撞发生所剩余的时间。对于跟驰和追尾型车辆交互，当后车相对于前车存在正向闭合速度时，TTC 可由纵向间距与纵向闭合速度之比计算。TTC 越小，表示车辆留给驾驶人采取避险操作的时间越短，冲突紧迫程度越高[6-7]。

Minderhoud 和 Bovy 在传统 TTC 基础上进一步提出了扩展碰撞时间指标，用于描述车辆处于临界 TTC 状态的持续时间及其累积严重程度。这说明道路风险不仅与某一时刻的 TTC 数值有关，还与风险状态的持续性和时间变化过程有关，为本文后续利用时序网络学习风险累积和风险演化规律提供了理论联系[6]。

为避免远距离轻微接近造成风险虚高，引入即时风险时域阈值 $T_h$，并使用归一化指数衰减函数将 TTC 映射为追尾风险概率：

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

已有概率化交通冲突研究指出，车辆冲突严重程度可以在确定性运动学指标的基础上进一步表示为连续风险量，并通过车辆交互状态计算碰撞风险或风险暴露程度。本文使用归一化指数衰减函数将 TTC 映射到 $[0,1]$ 区间，使较短 TTC 对应较高风险，属于在交通冲突时间指标基础上构造连续风险表征的方法[8]。

### 4.2 侧向擦碰风险

侧向风险用于描述变道、并行靠近、匝道汇入等横向逼近情形。

传统一维 TTC 主要面向纵向跟驰和追尾场景，而车辆变道、并行靠近、交织和匝道汇入等交通行为同时包含纵向与横向运动。近年来，替代安全指标研究逐渐由一维车辆质点模型扩展到多维运动空间，通过车辆相对位置、相对速度、运动方向和未来空间占用关系识别侧向或交叉型碰撞风险。因此，本文分别计算纵向冲突时间和侧向冲突时间，能够对不同方向上的车辆接近过程进行分解，并适应变道、汇入和侧向擦碰等复杂车辆交互场景[9]。

侧向相对速度定义为：

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

多维替代安全指标研究表明，仅考虑纵向距离可能无法完整表示真实道路中的碰撞演化过程，而同时考虑车辆纵向运动与横向运动能够提高对复杂车辆冲突的描述能力。本文通过纵向距离门控限制侧向冲突计算范围，并根据横向距离与横向相对速度计算侧向冲突剩余时间，体现了多维交通冲突分析中联合考虑空间接近关系和运动方向的基本思想[9]。

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

道路区域的整体安全状态由区域内多个车辆及多个车辆交互共同构成。Saunier 和 Sayed 提出的自动化概率安全分析框架将车辆轨迹、潜在碰撞位置和道路使用者交互结合起来，支持对单次交互风险进行概率化表示，并进一步构造跨车辆、跨时间和跨空间的聚合安全指标。本文由单车安全概率的乘积计算区域瞬时事故概率，体现了由微观车辆对冲突向区域整体风险聚合的建模思想[8]。

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

交通冲突并非相互独立的单帧事件，而是车辆之间相对位置、速度、加速度和驾驶操作在连续时间内共同演化的结果。已有研究开始将 TTC、驾驶操作状态和其他替代安全指标组织为时间序列，通过深度表示学习识别交通冲突；另有研究利用视频轨迹提取的交通流、车辆速度、车辆间距和车道级交通参数，实现信号交叉口交通冲突的实时预测[10-11]。

进一步地，短时事故风险预测研究表明，交通冲突指标、交通流状态及其连续时间变化可以共同输入时序学习模型，用于学习风险的动态相关性以及由一般交通扰动向持续高风险状态演化的规律。Fu 等将交通冲突指标与带有时序建模能力的 LSTM 结构结合，用于信号交叉口动态短时事故风险预测，为本文将机理风险序列与区域交通状态特征输入时序网络提供了直接研究依据[12]。

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

时序网络可采用 LSTM、GRU 或 TCN。

LSTM 通过输入门、遗忘门、输出门和记忆单元控制历史信息的保留与更新，能够缓解传统循环神经网络在长序列训练中出现的梯度消失问题，适合学习风险序列中的时间依赖关系[13]。GRU 使用更新门和重置门构建更加紧凑的门控循环结构，同样能够对连续时序状态进行编码[14]。TCN 通过因果卷积、扩张卷积和残差连接扩大时间感受野，可在不使用循环递推的条件下建模长短期序列依赖。因此，LSTM、GRU 和 TCN 均可作为本文风险演化预测层的候选时序模型[13-15]。

以 LSTM 为例，其递推过程为：

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

弱监督学习允许使用领域规则、启发式函数、知识模型或其他非人工逐样本标注方式生成训练标签。在真实事故和人工险情标签不足的情况下，可以将机理模型输出及其阈值规则作为标签生成函数，快速构建训练样本。相关研究表明，弱监督方法能够整合多个具有不同准确率和相关性的规则信号，为后续预测模型提供训练数据。因此，本文依据未来时域内的机理风险峰值构造弱监督标签，具有弱监督学习和数据编程方法的理论依据[16]。

二分类训练目标采用加权交叉熵，以缓解高危样本稀少带来的类别不平衡：

$$
\mathcal{L}=
-w_1Y_r(t)\log \widehat{Y}_r(t)
-w_0\left(1-Y_r(t)\right)
\log\left(1-\widehat{Y}_r(t)\right)
$$

其中，$w_1$ 与 $w_0$ 分别为正负样本权重。

在事故风险和高危冲突预测任务中，正常交通状态通常远多于高危状态，训练数据容易形成明显的类别不平衡。根据类别频率或有效样本数量对损失函数进行重新加权，可以提高少数类别误分类产生的训练代价，降低多数安全样本对模型优化过程的支配作用。因此，本文采用正负样本加权交叉熵，以增强模型对稀少高危样本的学习能力[17]。

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

# References

[1] Redmon, J., Divvala, S., Girshick, R., & Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 779-788. DOI: `10.1109/CVPR.2016.91`.

[2] Zhang, Y., Sun, P., Jiang, Y., Yu, D., Weng, F., Yuan, Z., Luo, P., Liu, W., & Wang, X. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. *Computer Vision – ECCV 2022*, Lecture Notes in Computer Science, 13682, 1-21. DOI: `10.1007/978-3-031-20047-2_1`.

[3] Zhang, Z. (2000). A Flexible New Technique for Camera Calibration. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 22(11), 1330-1334. DOI: `10.1109/34.888718`.

[4] Sochor, J., Juránek, R., & Herout, A. (2017). Traffic Surveillance Camera Calibration by 3D Model Bounding Box Alignment for Accurate Vehicle Speed Measurement. *Computer Vision and Image Understanding*, 161, 87-98. DOI: `10.1016/j.cviu.2017.05.015`.

[5] Savitzky, A., & Golay, M. J. E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. *Analytical Chemistry*, 36(8), 1627-1639. DOI: `10.1021/ac60214a047`.

[6] Minderhoud, M. M., & Bovy, P. H. L. (2001). Extended Time-to-Collision Measures for Road Traffic Safety Assessment. *Accident Analysis & Prevention*, 33(1), 89-97. DOI: `10.1016/S0001-4575(00)00019-1`.

[7] Laureshyn, A., Svensson, Å., & Hydén, C. (2010). Evaluation of Traffic Safety, Based on Micro-Level Behavioural Data: Theoretical Framework and First Implementation. *Accident Analysis & Prevention*, 42(6), 1637-1646. DOI: `10.1016/j.aap.2010.03.021`.

[8] Saunier, N., & Sayed, T. (2008). A Probabilistic Framework for Automated Analysis of Exposure to Road Collisions. *Transportation Research Record: Journal of the Transportation Research Board*, 2083(1), 96-104. DOI: `10.3141/2083-11`.

[9] Li, S., Anis, M., Lord, D., Zhang, H., Zhou, Y., & Ye, X. (2024). Beyond 1D and Oversimplified Kinematics: A Generic Analytical Framework for Surrogate Safety Measures. *Accident Analysis & Prevention*, 204, 107649. DOI: `10.1016/j.aap.2024.107649`.

[10] Lu, J., Grembek, O., & Hansen, M. (2022). Learning the Representation of Surrogate Safety Measures to Identify Traffic Conflict. *Accident Analysis & Prevention*, 174, 106755. DOI: `10.1016/j.aap.2022.106755`.

[11] Zhang, G., Jin, J., Chang, F., & Huang, H. (2024). Real-Time Traffic Conflict Prediction at Signalized Intersections Using Vehicle Trajectory Data and Deep Learning. *International Journal of Transportation Science and Technology*. DOI: `10.1016/j.ijtst.2024.10.009`.

[12] Fu, C., Lu, Z., Liu, H., & Wumaierjiang, A. (2025). Dynamic Short-Term Crash Risk Prediction from Traffic Conflicts at Signalized Intersections with Emerging Mixed Traffic Flow: A Novel Conflict Indicator. *Accident Analysis & Prevention*, 219, 108065. DOI: `10.1016/j.aap.2025.108065`.

[13] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780. DOI: `10.1162/neco.1997.9.8.1735`.

[14] Cho, K., van Merriënboer, B., Gülçehre, Ç., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning Phrase Representations Using RNN Encoder–Decoder for Statistical Machine Translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing*, 1724-1734. DOI: `10.3115/v1/D14-1179`.

[15] Bai, S., Kolter, J. Z., & Koltun, V. (2018). An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling. *arXiv preprint arXiv:1803.01271*. DOI: `10.48550/arXiv.1803.01271`.

[16] Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., & Ré, C. (2017). Snorkel: Rapid Training Data Creation with Weak Supervision. *Proceedings of the VLDB Endowment*, 11(3), 269-282. DOI: `10.14778/3157794.3157797`.

[17] Cui, Y., Jia, M., Lin, T. Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 9268-9277. DOI: `10.1109/CVPR.2019.00949`.

from copy import deepcopy
from pathlib import Path
import sys

sys.path.insert(0, r"D:\LDTP\report_formula_deps")
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from docx.text.paragraph import Paragraph
from latex2mathml.converter import convert
from lxml import etree

SOURCE = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_原理补充与语言优化版.docx")
OUTPUT = Path(r"C:\Users\pc\Desktop\报告3.5节 多目标算法内容_方法机理扩充版.docx")
doc = Document(SOURCE)
transform = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))
template = next(p for p in doc.paragraphs if p.text.startswith("气动计算在联合优化中需要反复执行"))

# Match the documented constraint aggregation to the implementation while retaining the paragraph formatting.
p = next(p for p in doc.paragraphs if p.text.startswith("候选先按任务、能源和飞行约束判定可行性"))
old = "超限方案按经尺度换算的约束超限程度排序。空速偏差、能源不足和姿态偏差分别按对应物理尺度换算为无量纲数值后合计，使优化更新可以追溯候选受到哪类限制。"
new = "超限方案按约束函数的正值之和比较。各项约束分别记录超限方向与程度，可据此追溯候选受到的限制；汇总值由各项原始约束值构成，其大小与各项约束采用的单位有关。"
hits = 0
for node in p._p.xpath(".//w:t"):
    if old in (node.text or ""):
        node.text = node.text.replace(old, new)
        hits += 1
if hits != 1:
    raise ValueError(f"constraint replacement matches: {hits}")

index = 0

def find(prefix):
    found = [p for p in doc.paragraphs if p.text.startswith(prefix)]
    if len(found) != 1:
        raise ValueError((prefix, len(found)))
    return found[0]

def insert(anchor, text=None, latex=None):
    global index
    node = OxmlElement("w:p")
    if template._p.pPr is not None:
        node.append(deepcopy(template._p.pPr))
    anchor._p.addnext(node)
    p = Paragraph(node, anchor._parent)
    if text is not None:
        r = p.add_run(text)
        r.font.name = "宋体"
        r._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        r.font.size = Pt(10.5)
    else:
        index += 1
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Pt(0)
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(4)
        node_math = transform(etree.fromstring(convert(latex).encode("utf-8"))).getroot()
        if node_math.tag == qn("m:oMathPara"):
            for child in list(node_math):
                p._p.append(child)
        else:
            p._p.append(node_math)
        r = p.add_run(f"    （3.5-D{index}）")
        r.font.name = "Times New Roman"
        r.font.size = Pt(10.5)
    return p

def block(prefix, entries):
    cursor = find(prefix)
    for kind, data in entries:
        cursor = insert(cursor, text=data if kind == "p" else None,
                        latex=data if kind == "e" else None)

block("电池SOC指标包括任务过程中的最低值和终端值", [
    ("p", "能量评价以任务时间历程为基础。将燃油质量流量对时间积分，得到任务消耗的燃油质量；与燃油低热值相乘，得到燃油化学能消耗。电池部分采用端口功率与充、放电效率区分储能变化和对外传递能量。设放电功率和充电功率分别以非负量Pdis、Pch表示，ηdis、ηch分别为对应转换效率，则"),
    ("e", r"E_{\mathrm{dis,cell}}=\int_{0}^{T}\frac{P_{\mathrm{dis}}(t)}{\eta_{\mathrm{dis}}}\,dt,\quad E_{\mathrm{dis,bus}}=\int_{0}^{T}P_{\mathrm{dis}}(t)\,dt"),
    ("e", r"E_{\mathrm{ch,bus}}=\int_{0}^{T}P_{\mathrm{ch}}(t)\,dt,\quad E_{\mathrm{ch,cell}}=\int_{0}^{T}\eta_{\mathrm{ch}}P_{\mathrm{ch}}(t)\,dt"),
    ("p", "式中cell对应电池储能侧，bus对应端口侧。放电时储能减少量大于端口输出量；充电时端口输入量大于储能增加量。按这两个过程汇总，电池净储能减少量、转换损失与任务能量目标分别为"),
    ("e", r"\Delta E_{\mathrm{bat}}=E_{\mathrm{dis,cell}}-E_{\mathrm{ch,cell}}"),
    ("e", r"E_{\mathrm{loss}}=(E_{\mathrm{dis,cell}}-E_{\mathrm{dis,bus}})+(E_{\mathrm{ch,bus}}-E_{\mathrm{ch,cell}})"),
    ("e", r"J_E=H_{\mathrm{LHV}}\int_{0}^{T}\dot m_f(t)\,dt+\Delta E_{\mathrm{bat}}"),
    ("p", "其中Hₗₕᵥ为燃油低热值，ṁf为燃油质量流量。上式以电池净储能减少量进入任务能量目标，转换损失已经体现在储能侧与端口侧的能量差中。能量计算应统一功率、时间与能量单位；任务末端储备仍通过燃油余量和SOC约束单独检查。"),
    ("p", "动力分配影响各能量项的时间分布。发动机承担较多负荷时，燃油流量和电池充放电需求随之变化；电池承担辅助负荷时，端口输出、储能减少及终端SOC共同变化。比较方案时，将总能量目标与燃油、电池净消耗和转换损失同时列出，有助于识别能耗变化对应的供能环节。"),
])

block("响应时间指标由扰动恢复时间、动力模式切换时间和恢复约束惩罚组成", [
    ("p", "跟踪指标首先将不同物理量的偏差转换为可比较的无量纲量。设高度、速度和俯仰角误差以及俯仰角速度分别为eₕ、eᵥ、eθ和q，对应尺度为sh、sv、sθ和sq，则单项归一化量可统一记为"),
    ("e", r"\tilde e_h=e_h/s_h,\quad \tilde e_v=e_v/s_v,\quad \tilde e_\theta=e_\theta/s_\theta,\quad \tilde q=q/s_q"),
    ("p", "时间平均项对四项归一化量的平方加权积分，衡量整个任务中的持续偏差；峰值项取高度、速度和俯仰角三个归一化误差绝对值在任务区间内的最大值，保留短时较大偏差的信息。设相应权重为wi和wp，综合形式为"),
    ("e", r"J_{\mathrm{track}}=\frac{1}{T}\int_0^T\sum_{i=1}^{4}w_i\tilde e_i^2(t)\,dt+w_p e_{\mathrm{peak}}"),
    ("e", r"e_{\mathrm{peak}}=\max_{t\in[0,T]}\max\{|\tilde e_h(t)|,|\tilde e_v(t)|,|\tilde e_\theta(t)|\}"),
    ("p", "平均项与峰值项从不同时间尺度描述控制表现：持续的小幅偏差会累积到平均项，持续时间较短但幅值较大的偏差通过峰值项进入评价。归一化尺度体现各状态量的允许偏差或评价尺度，同一组方案采用一致尺度和权重。"),
    ("p", "恢复时间的判定同时考虑偏差幅值和保持时间。设扰动结束时刻为td，受扰轨迹与参考轨迹的状态差为Δz，保持时长为Th，则首先寻找扰动结束后满足连续保持条件的最早时刻τ；恢复时间由该时刻与扰动结束时刻之差给出"),
    ("e", r"t_{\mathrm{rec}}=\min_{\tau\geq t_d}\{\tau-t_d:\|\Delta z(t)\|\leq\varepsilon,\ t\in[\tau,\tau+T_h]\}"),
    ("p", "状态差的具体构成与阈值依任务评价设置确定。持续保持条件使恢复判定对应一段稳定运行区间；在给定评价时域内未满足恢复条件时，按评价模型设置的恢复失败惩罚处理。模式切换时间则描述功率指令改变后实际输出建立所需时间，两者与相应权重共同构成响应目标。"),
])

block("能源、跟踪和响应之间的冲突来自协同模型中的同一功率安排", [
    ("p", "从统一优化表达看，设计向量中的气动参数影响升阻力与所需推进功率，动力参数决定部件的可用功率和能源容量，控制参数影响状态偏差及功率指令。任务仿真将这些参数映射为连续状态与功率历程，性能指标和约束由同一历程提取，因此不同目标之间的联系可以沿着共同的物理状态追踪。"),
    ("p", "约束函数按满足条件小于或等于零表示。对飞行或动力变量的上限，可用全任务最大值与上限之差构造约束；对终端能源储备下限，可用最低要求与终端剩余量之差构造约束"),
    ("e", r"g_{\mathrm{upper}}(x)=\max_{t\in[0,T]}z(t;x)-z_{\max}"),
    ("e", r"g_{\mathrm{end}}(x)=E_{\min}-E(T;x)"),
    ("p", "当前评价实现对各约束函数的正值直接求和，构成不可行候选之间的比较量；可行候选的各项正值均为零。该比较量汇总了不同约束的超限程度，各项函数的量纲和数值尺度会影响其相对贡献，因此应同时保留各项约束结果以解释主要限制来源。"),
    ("e", r"V(x)=\sum_{j=1}^{n_g}\max\{0,g_j(x)\}"),
    ("p", "目标量用于比较满足约束后的性能取舍，约束量用于检查部件能力、能源储备和飞行条件。目标归一化采用固定参考值和尺度，保证方案决策中不同目标的数值具有一致的比较口径；该处理与约束函数的可行性判定分别进行。"),
])

block("尾迹效率因子的计算采用升力系数平方与诱导阻力系数之间的关系", [
    ("p", "代理模型的两个输出分别进入升力和阻力关系。在线性升力近似的适用范围内，升力系数可用零升力迎角附近的斜率描述；尾迹效率因子与展弦比共同影响诱导阻力项"),
    ("e", r"C_L\simeq C_{L_\alpha}(\alpha-\alpha_0),\qquad C_{D_i}\simeq \frac{C_L^2}{\pi ARe}"),
    ("p", "式中CLα为升力斜率，α为迎角，α₀为零升力迎角，AR为展弦比，e为尾迹效率因子。相同飞行状态和参考面积下，升力系数影响升力平衡所需迎角，诱导阻力系数影响气动阻力与推进功率需求；这两类气动量因而分别进入飞行状态和能源消耗的计算链条。"),
])
block("Kriging及高斯过程利用样本间相关性预测新构型", [
    ("p", "以单个气动输出为例，设构型特征向量为u，训练构型及对应气动量组成样本集。高斯过程将该气动量表示为平均趋势与具有相关结构的随机偏差之和；对任意两个构型，协方差函数描述其气动偏差之间的相关程度"),
    ("e", r"y(u)=m(u)+f(u),\qquad f(u)\sim\mathcal{GP}(0,k(u,u'))"),
    ("p", "给定训练输出向量y、均值向量m和训练特征间的协方差矩阵K，预测构型u*与训练样本的协方差组成向量k*。在高斯观测噪声模型下，对训练数据进行条件化得到预测均值和潜在函数预测方差"),
    ("e", r"\hat y_*=m(u_*)+k_*^{T}(K+\sigma_n^2I)^{-1}(y-m)"),
    ("e", r"s_*^2=k(u_*,u_*)-k_*^{T}(K+\sigma_n^2I)^{-1}k_*"),
    ("p", "其中σn²表示观测噪声方差，I为单位矩阵。预测均值由训练输出相对趋势项的偏差及其与新构型的相关程度共同确定；预测方差反映在所选相关函数和训练样本条件下对潜在函数的估计不确定程度。两个气动输出分别建立模型，使各自的相关尺度和预测误差可以独立描述。"),
    ("p", "对径向基相关函数，可为不同输入特征设置独立长度尺度。设第j个特征的长度尺度为ℓj，幅值参数为σf²，则构型间相关性随各维特征距离变化"),
    ("e", r"k(u,u')=\sigma_f^2\exp\left[-\frac{1}{2}\sum_{j=1}^{d}\frac{(u_j-u'_j)^2}{\ell_j^2}\right]"),
    ("p", "在其余条件相同的情况下，较短的长度尺度表示模型对该特征方向的变化更敏感。输入特征先按训练样本的尺度处理，再由样本确定相关函数参数；预测后将输出转换回原有气动量单位。Matérn相关函数同样通过距离刻画样本间联系，其平滑程度由核函数形式决定。"),
])

block("推荐方案输出几何、动力配置与控制参数，并附带目标值和能源、速度裕度", [
    ("p", "多目标搜索形成的是一组性能取舍各异的可行候选，方案决策在该集合内引入任务偏好。对目标fm采用固定参考值fref,m和正尺度sm进行归一化，再以非负权重wm求得综合评分"),
    ("e", r"\tilde f_m(x)=\frac{f_m(x)-f_{\mathrm{ref},m}}{s_m},\quad S_w(x)=\sum_{m=1}^{3}w_m\tilde f_m(x)"),
    ("e", r"x_w^*=\arg\min_{x\in\mathcal P_{\mathrm{screen}}}S_w(x),\qquad w_m\geq0,\quad \sum_{m=1}^{3}w_m=1"),
    ("p", "其中筛选集合由满足任务约束并通过能源储备等附加条件的候选组成。归一化用于统一目标尺度，权重用于表达决策偏好：提高某一目标的权重，会提高该目标在综合评分中的相对作用。评分在已有筛选集合上进行，候选集合的质量由前序搜索和可行性评价确定。"),
    ("p", "权重扫描通过改变目标权重观察推荐方案的变化。相同方案在一段权重范围内持续入选，表明该方案对这一范围内的偏好变化具有稳定的推荐结果；当推荐方案发生变化时，可比较两方案的原始目标值及能源、飞行约束裕度，明确性能改善与代价。"),
    ("p", "对筛选后的候选开展完整任务复算时，输入包括该候选的几何、动力和控制参数；评价输出包括轨迹、功率、燃油与电池状态，以及对应目标值和约束量。复算结果与搜索评价采用相同的指标定义，便于追踪设计参数变化如何传递到任务性能。"),
])

block("算法评价从解集质量、可行方案比例和计算耗时三个方面展开", [
    ("p", "超体积采用统一归一化目标空间中的参考点r，度量可行非支配解集支配区域的大小。对于最小化问题，每个目标点与参考点构成一个区间集合；对全部候选对应区域取并集，再计算其体积"),
    ("e", r"HV(\mathcal P;r)=\operatorname{vol}\left(\bigcup_{x\in\mathcal P}\prod_{m=1}^{3}[\tilde f_m(x),r_m]\right)"),
    ("p", "式中P为用于计算的可行非支配解集，参考点在所计入目标点的较差一侧。并集运算使不同方案重叠覆盖的区域只计算一次；目标点向更优方向移动或补充新的非重叠覆盖区域，都会改变超体积。比较不同算法时，目标归一化、参考点和候选评价口径保持一致。"),
    ("p", "终代可行率按最终种群中满足全部约束的候选数与种群总数之比计算。超体积描述可行解集覆盖的目标区域，可行率描述终代候选满足约束的比例，计算耗时描述整个求解过程的时间开销；三项指标共同用于评价搜索结果和计算代价。"),
    ("e", r"R_{\mathrm{feas}}=N_{\mathrm{feas}}/N_{\mathrm{pop}}"),
])

block("控制参数敏感性沿供能过程分析", [
    ("p", "秩相关分析先将各样本的设计变量和性能指标分别转换为秩，再计算两组秩的相关系数。对存在相同取值的样本使用平均秩，Spearman系数可按两组秩的协方差与标准差计算"),
    ("e", r"\rho_s=\frac{\operatorname{cov}(R_x,R_y)}{\sigma_{R_x}\sigma_{R_y}}"),
    ("p", "式中Rx和Ry分别为变量x与指标y的样本秩。系数的符号表示样本中两者总体变化方向，绝对值反映单调关联的强弱。气动构型、动力参数和控制设置可能同时影响结果，因此相关关系用于确定值得进一步分析的变量；具体作用机理再结合单项参数变化和任务状态历程解释。"),
    ("p", "单项参数分析以统一基准方案为起点，逐项改变参数并重新评价，其余输入保持一致。将目标变化与推进功率、电池状态、燃油消耗和跟踪误差历程对应，可以区分参数变化对气动需求、供能过程和控制响应的影响；对不同指标采用相同的基准和评价口径，以保持结果的可比性。"),
])

doc.save(OUTPUT)
print("saved:", OUTPUT, "new editable equations:", index)

from pathlib import Path
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn

root = Path(r"C:\Users\pc\Desktop\PTDL中期汇报_第5至8页素材")
root.mkdir(exist_ok=True)
images = root / "报告原图"
images.mkdir(exist_ok=True)
report = Document(r"C:\Users\pc\Desktop\PTDL 1.2 2.2节报告 (1).docx")
figs = {6:"多目标搜索与候选方案评价",7:"气动代理模型的建立与应用",
        8:"气动代理独立验证结果对照",10:"四算法的收敛过程与重复运行分布",
        12:"推进系统与飞行控制协同设计流程"}
for number, name in figs.items():
    for i, p in enumerate(report.paragraphs[:-1]):
        ids = p._p.xpath(".//a:blip/@r:embed")
        caption = report.paragraphs[i+1].text.strip()
        if ids and re.match(fr"^图\s*{number}(?!\d)\s", caption):
            part = report.part.related_parts[ids[0]]
            ext = Path(str(part.partname)).suffix
            (images / f"图{number}_{name}{ext}").write_bytes(part.blob)
            break
    else:
        raise ValueError(f"未定位报告图{number}")

slides = [
{
 "page": 5, "title": "多目标性能优化指标体系与约束条件构建",
 "start": "面向长航时无人机的能源利用与飞行控制需求，构建多目标性能指标及任务约束体系。",
 "end": "建立了设计变量、性能指标与任务约束之间的评价关系，为多目标协同优化提供基础。",
 "layout": "中间框按左、中、右三栏排列，栏宽约为 26%、43%、31%。左栏展示设计变量，中栏突出三个优化目标，右栏展示任务约束；栏与栏之间用细箭头连接。每栏采用深蓝标题、下方简短文字和配图，保持白底。",
 "blocks":[
  ("左栏  设计变量", "可直接上屏的文字：气动构型 / 动力配置 / 控制参数。图形采用机翼轮廓、动力单元和控制指令三个工程化小图标，竖向排列。"),
  ("中栏  优化目标", "可直接上屏的文字：任务能量消耗——燃油化学能与电池净耗能；飞行跟踪误差——高度、速度与姿态偏差；响应时间指标——扰动恢复与模式切换。三个目标纵向排列，适当放大。"),
  ("右栏  任务约束", "可直接上屏的文字：部件功率与电流；燃油与电池储备；飞行状态与起降条件。栏底用小字注明“燃油、电池效率等为辅助评价指标”。")
 ],
 "figure": "首选报告表7“优化目标与其他评价量的作用”和表9“协同优化的主要约束关系”提炼成图文关系；表格内容较多，不建议整表截图。报告没有对应的完整构图，需补充方法示意。",
 "prompt": "绘制科研项目汇报用的三栏工程方法图，白底、深蓝色标题、少量蓝绿配色、中文大字号。左栏显示三类设计输入：气动构型、动力配置、控制参数；中栏显示三个正式优化目标：任务能量消耗、飞行跟踪误差、响应时间指标；右栏显示约束：部件功率与电流、燃油和电池储备、飞行状态与起降条件。用箭头表示设计参数经任务评价形成目标和约束。使用准确简洁的工程图标，避免装饰性背景；不添加公式、数值或虚构试验数据。"
},
{
 "page": 6, "title": "多目标优化控制算法发展与代理模型方法研究",
 "start": "针对飞行性能、推进系统与控制策略的耦合关系，采用多种群体搜索方法开展多目标优化。",
 "end": "形成了基于统一任务评价的多算法搜索与可行方案筛选方法。",
 "layout": "中间框采用上中下三层。上层并排放四个算法名称及各自的一行机制，中层大幅放报告图6，下层用三项短语说明候选评价和方案保留。算法名称与流程节点对齐。",
 "blocks":[
  ("上层  四类搜索算法", "可直接上屏的文字：NSGA-II｜非支配排序；MOEA/D｜目标分解与邻域搜索；MOPSO｜历史经验与粒子更新；混合算法｜遗传重组与粒子更新。四项同等宽度横向排列。"),
  ("中层  搜索与评价", "使用报告图6作为主体。展示“候选设计生成 → 飞行与动力联合评价 → 目标及约束计算 → 可行方案保留 → 继续搜索”的关系。可根据版面裁掉图题，保留箭头及主要节点。"),
  ("下层  共用评价与输出", "可直接上屏的文字：统一任务模型 / 三项目标与约束 / 可行非支配方案集。注意四种算法共享评价条件，图中线条表达不同搜索机制汇入同一评价。")
 ],
 "figure": "优先使用报告图6“多目标搜索与候选方案评价”，原图已另存至“报告原图”文件夹。图6若缩放后文字难以辨认，按其逻辑在PPT中手工重排文字和箭头。",
 "prompt": ""
},
{
 "page": 7, "title": "多目标优化控制算法发展与代理模型方法研究",
 "start": "基于参数化构型及其气动计算结果，建立气动代理模型，实现候选方案的快速评价。",
 "end": "气动代理模型建立了构型参数与气动特性之间的预测关系，提高了多目标优化中的候选评价效率。",
 "layout": "中间框左右两栏，左栏约占48%，右栏约占52%。左栏用报告图7展示样本生成与模型建立；右栏放报告图8的独立验证对比。两个小标题置于各栏上方，图注与数据紧贴图片。",
 "blocks":[
  ("左栏  气动代理建模", "可直接上屏的文字：参数化构型 → 气动计算 → 气动样本 → 代理模型。图下补充“分别预测升力斜率与尾迹效率因子”。构型由OpenVSP建立，气动样本由VSPAERO计算。"),
  ("右栏  独立构型验证", "使用图8和表11。可直接上屏的文字：升力斜率 R² 0.81 → 0.88；尾迹效率因子 R² 0.71 → 0.89。两项均来自报告中同一独立构型集的原始尺寸模型与改进模型对比。"),
  ("底部说明", "可直接上屏的文字：代理模型用于候选快速评价，推荐方案的气动特性再由VSPAERO复算。对应关系用一条短箭头表示。")
 ],
 "figure": "优先使用报告图7“气动代理模型的建立与应用”、图8“气动代理独立验证结果对照”，两张原图已提取。以本报告表11的结果为准，不与其他批次模型精度混用。",
 "prompt": ""
},
{
 "page": 8, "title": "多目标优化控制仿真验证与算法性能评估",
 "start": "结合全任务仿真与算法性能对比，开展飞推控一体化协同设计方法验证。",
 "end": "形成面向长航时无人机的飞推控一体化协同设计与多目标优化仿真验证方法体系。",
 "layout": "中间框左右两栏，左栏约占45%，右栏约占55%。左栏放协同设计与复算流程，右栏放四算法收敛或结果分布。图下保留三项紧凑的评价文字；全页不使用平台截图和平台标题。",
 "blocks":[
  ("左栏  协同设计与任务复算", "使用报告图12。可直接上屏的文字：任务与模型确定 → 控制参数整定 → 多目标方案搜索 → 可行方案复算。复算关注飞行轨迹、能源消耗与末端状态。"),
  ("右栏  算法性能比较", "使用报告图10，优先保留清晰可读的收敛曲线；根据表13在图下放三项短语：解集质量 / 终代可行率 / 计算耗时。图示的统计范围和含义随原图保留。"),
  ("结果摘录", "可直接上屏的文字：MOPSO平均超体积0.97，终代可行率80.4%；混合算法终代可行率100%，平均耗时14.5 min。两项为报告中的同预算比较结果，各自对应不同评价维度。")
 ],
 "figure": "优先使用报告图12“推进系统与飞行控制协同设计流程”、图10“四算法的收敛过程与重复运行分布”和表13“共同计算预算下的四类算法比较”。相关原图已提取。",
 "prompt": ""
}]

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin, sec.bottom_margin = Inches(.62), Inches(.6)
sec.left_margin, sec.right_margin = Inches(.8), Inches(.8)
normal = doc.styles["Normal"]
normal.font.name = "宋体"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor.from_string("202020")
title = doc.styles["Title"]
title.font.name = "微软雅黑"
title._element.rPr.rFonts.set(qn("w:eastAsia"), "微软雅黑")
title.font.size = Pt(16)
title.font.color.rgb = RGBColor(0, 0, 0)
for border in title._element.get_or_add_pPr().findall(qn("w:pBdr")):
    title._element.get_or_add_pPr().remove(border)

def para(label, text, after=5):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.14
    p.add_run(label + "  ").bold = True
    p.add_run(text)

for idx, s in enumerate(slides):
    if idx:
        doc.add_page_break()
    if not idx:
        p = doc.add_paragraph("中期汇报第5至8页 图文大纲", style="Title")
        p.paragraph_format.space_after = Pt(3)
        p = doc.add_paragraph("依据《PTDL 1.2 2.2节报告 (1)》的2.2节，沿用报告原图图号")
        p.paragraph_format.space_after = Pt(8)
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run(f"第{s['page']}页  {s['title']}")
    r.bold = True
    r.font.size = Pt(14)
    para("开头总领句", s["start"])
    para("中间框总体布局", s["layout"])
    for heading, body in s["blocks"]:
        para(heading, body)
    para("插图来源", s["figure"])
    para("结尾总结句", s["end"])

doc.add_page_break()
p = doc.add_paragraph()
p.add_run("补充配图提示词").bold = True
para("第5页  指标与约束", slides[0]["prompt"])
para("第6页  算法搜索流程", "以报告图6为依据重绘一幅科研汇报方法图。上方并列表示NSGA-II、MOEA/D、MOPSO和遗传粒子混合算法；中部展示飞机几何、动力、控制候选进入气动、推进和飞行任务联合评价；右侧区分任务能量、飞行跟踪、响应时间以及约束检查；可行候选通过非支配筛选反馈到下一次搜索。用精细工程图标、清晰连线和适量留白表达过程，白底、深蓝与蓝绿色、大字号。保持四种算法共用同一评价模型，不添加未经提供的数据、算法结果或复杂功能。")
para("第7页  气动代理建模", "以报告图7为依据绘制横向科研方法图：参数化飞机几何、试验设计选点、VSPAERO气动计算、样本质量检查、升力斜率和尾迹效率分别建模、独立构型验证、优化候选快速预测与重点方案气动复算。用飞机几何与科学计算图标表达各环节，保留分目标建模与验证顺序。白底、深蓝绿配色，适合PPT投影，不添加数值或虚构算例。右侧的独立测试结果使用报告图8，不能由模型生成。")
para("第8页  协同设计方法", "以报告图12为依据绘制科研汇报用的协同设计方法图，内容包括任务与指标、气动动力飞控协同模型、控制参数选择、多目标搜索、任务偏好筛选、完整任务复算；以箭头表示设计信息传递与迭代。用飞机、推进系统、轨迹、目标与约束等工程元素辅助理解，避免空洞装饰。白底、深蓝与蓝绿色、文字清晰，不展示不存在的数据或平台界面。右侧算法对比使用报告图10。")

footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run("第5至8页制作参考")
target = root / "第5至8页_图文制作大纲.docx"
doc.save(target)
print(target)
for f in images.iterdir():
    print(f.name)

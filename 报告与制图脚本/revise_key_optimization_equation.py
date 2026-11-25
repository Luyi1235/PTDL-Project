from copy import deepcopy
from lxml import etree
from docx import Document
from docx.oxml.ns import qn

src = r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_公式示意图版.docx"
out = r"C:\Users\pc\Desktop\多目标自适应混合算法关键技术_优化问题公式修订版.docx"
doc = Document(src)
p = doc.paragraphs[5]
assert len(p._p.xpath(".//m:oMath")) == 1
mathml = r"""
<math xmlns="http://www.w3.org/1998/Math/MathML">
 <mtable columnalign="left left" columnspacing="2em">
  <mtr>
   <mtd><mtext>minimize</mtext></mtd>
   <mtd><mrow><mi>J</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mo>[</mo>
     <msub><mi>J</mi><mi>e</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo>
     <msub><mi>J</mi><mi>s</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>,</mo>
     <msub><mi>J</mi><mi>r</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo>
     <msup><mo>]</mo><mi>T</mi></msup></mrow></mtd>
  </mtr>
  <mtr>
   <mtd><mtext>with respect to</mtext></mtd>
   <mtd><mrow><mi>x</mi><mo>∈</mo><mi>Ω</mi></mrow></mtd>
  </mtr>
  <mtr>
   <mtd><mtext>subject to</mtext></mtd>
   <mtd><mrow><msub><mi>g</mi><mi>j</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo><mo>≤</mo>
     <mn>0</mn><mo>,</mo><mspace width="0.75em"/>
     <mi>j</mi><mo>=</mo><mn>1</mn><mo>,</mo><mo>…</mo><mo>,</mo><mi>m</mi></mrow></mtd>
  </mtr>
 </mtable>
</math>
"""
transform = etree.XSLT(etree.parse(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL"))
result = transform(etree.fromstring(mathml.encode("utf-8"))).getroot()
replacement = result.find(qn("m:oMath")) if result.tag.endswith("}oMathPara") else result
for alignment in replacement.xpath('.//*[local-name()="mcJc"]'):
    alignment.set(qn("m:val"), "left")
old = p._p.xpath(".//m:oMath")[0]
old.getparent().replace(old, deepcopy(replacement))

explanation = doc.paragraphs[6]
assert explanation.text.startswith("式中，x为设计变量")
explanation.clear()
explanation.add_run(
    "式中，x为由构型、动力及任务控制参数组成的设计变量向量；"
    "Jₑ、Jₛ和Jᵣ分别为任务能源消耗、飞行跟踪误差及响应时间指标；"
    "Ω为设计变量的取值空间，各变量取值由相应上下限确定；"
    "gⱼ为第j项约束函数，m为约束数量。"
)
doc.save(out)
print(out)

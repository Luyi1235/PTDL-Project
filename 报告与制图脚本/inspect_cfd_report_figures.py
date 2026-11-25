from pathlib import Path
from docx import Document

doc = Document(r"C:\Users\pc\Desktop\【V3】附件3-技术进展报告总_交叉引用宋体修正版.docx")
out = Path(r"D:\LDTP\cfd_report_reference")
out.mkdir(exist_ok=True)
for number, relationship in [(20, "rId33"), (22, "rId35"), (25, "rId38"), (26, "rId39")]:
    (out / f"report_{number}.png").write_bytes(doc.part.related_parts[relationship].blob)
print(out)

from PIL import Image, ImageDraw, ImageFont

source = r"C:\Users\pc\AppData\Local\Temp\codex-clipboard-c3879af5-12cf-404a-9344-90bb22c49ce1.png"
output = r"C:\Users\pc\Desktop\PTDL_多目标指标与约束_动词标题版.png"
font_path = r"C:\Windows\Fonts\msyhbd.ttc"

image = Image.open(source).convert("RGB")
draw = ImageDraw.Draw(image)


def clear_from_column(box, sample_x):
    x0, y0, x1, y1 = box
    for y in range(y0, y1):
        color = image.getpixel((sample_x, y))
        draw.line((x0, y, x1, y), fill=color)


def fitted_text(box, text, color, max_size, centered=False):
    x0, y0, x1, y1 = box
    for size in range(max_size, 15, -1):
        font = ImageFont.truetype(font_path, size)
        bounds = draw.textbbox((0, 0), text, font=font)
        width = bounds[2] - bounds[0]
        height = bounds[3] - bounds[1]
        if width <= x1 - x0 and height <= y1 - y0:
            break
    x = x0 + (x1 - x0 - width) / 2 if centered else x0
    y = y0 + (y1 - y0 - height) / 2 - bounds[1]
    draw.text((round(x), round(y)), text, font=font, fill=color)


for box, sample_x, text in [
    ((48, 25, 590, 80), 38, "确定协同设计变量"),
    ((675, 25, 1340, 80), 665, "建立多目标性能指标"),
    ((1420, 25, 1965, 80), 1410, "设置任务约束条件"),
]:
    clear_from_column(box, sample_x)
    fitted_text(box, text, "white", 43, centered=True)

for box, text in [
    ((80, 113, 260, 153), "确定气动构型"),
    ((80, 280, 260, 320), "确定动力配置"),
    ((80, 452, 260, 492), "设置控制参数"),
]:
    clear_from_column(box, 245)
    fitted_text(box, text, "#073b82", 29)

for box, text in [
    ((687, 108, 964, 145), "1. 评估任务能量消耗"),
    ((687, 277, 964, 314), "2. 评估飞行跟踪误差"),
    ((687, 439, 964, 476), "3. 评估任务响应时间"),
]:
    clear_from_column(box, 680)
    fitted_text(box, text, "#063b7b", 27)

for box, text in [
    ((1450, 105, 1860, 142), "约束部件功率与电流"),
    ((1450, 239, 1860, 276), "约束燃油与电池储备"),
    ((1450, 377, 1860, 414), "约束飞行状态与起降条件"),
]:
    clear_from_column(box, 1444)
    fitted_text(box, text, "#073b82", 28)

image.save(output)
print(output)

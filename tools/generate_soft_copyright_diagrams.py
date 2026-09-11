# -*- coding: utf-8 -*-
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "soft_copyright_assets"


def font_path():
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def font(size):
    path = font_path()
    return ImageFont.truetype(path, size=size) if path else ImageFont.load_default()


def wrapped_text(draw, text, rect, font_obj, fill=(20, 32, 46), line_gap=8):
    x1, y1, x2, y2 = rect
    max_width = x2 - x1
    lines = []
    for raw_line in str(text).splitlines():
        line = ""
        for ch in raw_line:
            test = line + ch
            box = draw.textbbox((0, 0), test, font=font_obj)
            if box[2] - box[0] <= max_width or not line:
                line = test
            else:
                lines.append(line)
                line = ch
        if line:
            lines.append(line)

    sample = draw.textbbox((0, 0), "高速车辆轨迹规划", font=font_obj)
    line_height = sample[3] - sample[1] + line_gap
    total_height = line_height * len(lines) - line_gap
    y = y1 + max(0, (y2 - y1 - total_height) // 2)
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font_obj)
        x = x1 + (max_width - (box[2] - box[0])) // 2
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += line_height


def box(draw, xy, text, fill, outline, font_obj):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=2)
    wrapped_text(draw, text, (xy[0] + 18, xy[1] + 16, xy[2] - 18, xy[3] - 16), font_obj)


def arrow(draw, start, end, fill=(48, 83, 124), width=4):
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        sign = 1 if ex >= sx else -1
        points = [(ex, ey), (ex - sign * 16, ey - 9), (ex - sign * 16, ey + 9)]
    else:
        sign = 1 if ey >= sy else -1
        points = [(ex, ey), (ex - 9, ey - sign * 16), (ex + 9, ey - sign * 16)]
    draw.polygon(points, fill=fill)


def add_title(draw, title):
    draw.text((70, 45), title, font=font(34), fill=(13, 45, 77))


def make_development_flow():
    w, h = 1680, 1020
    img = Image.new("RGB", (w, h), (247, 250, 252))
    draw = ImageDraw.Draw(img)
    add_title(draw, "软件开发流程图")
    body = font(18)
    small = font(16)
    border = (82, 113, 150)
    colors = [
        (222, 238, 252),
        (255, 243, 213),
        (224, 245, 235),
        (236, 229, 250),
        (222, 238, 252),
        (255, 243, 213),
        (224, 245, 235),
        (236, 229, 250),
    ]
    steps = [
        "需求分析\n高速跟车/变道\n仿真训练与结果查看",
        "环境准备\nCARLA 0.9.9.2\nPython 依赖与工程目录",
        "模块设计\nGym 环境\nFrenet 规划\n车辆控制",
        "算法配置\nDDPG/TRPO/A2C\nPPO2/SAC\nYAML 参数",
        "编码实现\nrun.py 主入口\ncarla_gym 与 agents 模块",
        "训练与测试\n模型保存\n日志与轨迹 CSV",
        "可视化验证\n奖励曲线\n轨迹图/对比图\nDashboard 导出",
        "文档与交付\n软著说明书\n版本记录\n人工确认事项",
    ]
    positions = [
        (80, 155, 390, 295),
        (500, 155, 810, 295),
        (920, 155, 1230, 295),
        (1300, 155, 1610, 295),
        (1300, 545, 1610, 685),
        (920, 545, 1230, 685),
        (500, 545, 810, 685),
        (80, 545, 390, 685),
    ]
    for i, (xy, text) in enumerate(zip(positions, steps)):
        box(draw, xy, text, colors[i], border, body)

    def draw_step_number(index, xy):
        cx = (xy[0] + xy[2]) // 2
        draw.ellipse((cx - 28, xy[1] - 74, cx + 28, xy[1] - 18), fill=(13, 82, 128))
        label = str(index + 1)
        label_box = draw.textbbox((0, 0), label, font=font(24))
        draw.text((cx - (label_box[2] - label_box[0]) // 2, xy[1] - 68), label, font=font(24), fill=(255, 255, 255))

    for i in range(3):
        arrow(draw, ((positions[i][2], 225)), ((positions[i + 1][0] - 12, 225)))
    arrow(draw, ((1455, 295)), ((1455, 545)))
    for i in range(4, 7):
        arrow(draw, ((positions[i][0], 615)), ((positions[i + 1][2] + 12, 615)))

    for i, xy in enumerate(positions):
        draw_step_number(i, xy)

    note_boxes = [
        (210, 795, 570, 935, "输入依据\nREADME / requirements.txt\ntools/cfgs / run.py\ncarla_gym / agents\nlogs / outputs"),
        (660, 795, 1020, 935, "核查原则：代码优先；无法确认的功能写为可配置或列入人工确认事项"),
        (1110, 795, 1470, 935, "交付内容：修订版 Word 说明书、流程图、架构图、截图图题和修改摘要"),
    ]
    for xy in note_boxes:
        box(draw, xy[:4], xy[4], (255, 255, 255), (188, 204, 220), small)

    out = OUT_DIR / "software_development_flow.png"
    img.save(out)
    return out


def make_architecture():
    w, h = 1680, 1060
    img = Image.new("RGB", (w, h), (247, 250, 252))
    draw = ImageDraw.Draw(img)
    add_title(draw, "软件架构图")
    body = font(18)
    border = (82, 113, 150)
    blue = (222, 238, 252)
    green = (224, 245, 235)
    gold = (255, 243, 213)
    purple = (236, 229, 250)

    boxes = {
        "entry": ((80, 150, 410, 295), "入口层\nrun.py\n命令行参数解析\n配置加载", blue),
        "dashboard": ((80, 390, 410, 535), "可视化层\npygame_dashboard.py\nPygame/Tk Dashboard\nmonitor_plot.py", purple),
        "cfg": ((525, 150, 855, 295), "配置层\ntools/cfgs/*.yaml\n算法/环境/奖励/交通参数", gold),
        "rl": ((525, 390, 855, 535), "算法层\nStable Baselines\nDDPG/TRPO/A2C/PPO2/SAC", green),
        "env": ((970, 150, 1325, 295), "环境层\ncarla_gym/envs\nCarlaGymEnv-v1/v2\n状态、动作、奖励", blue),
        "planner": ((970, 390, 1325, 535), "规划层\nagents/local_planner\nFrenet 候选轨迹\n选定轨迹输出", green),
        "control": ((970, 615, 1325, 760), "控制层\nlow_level_controller\nPID 控制器\nIDM 跟驰模型", gold),
        "carla": ((970, 835, 1325, 980), "仿真层\nCARLA 服务端\nTown04 场景\n交通流与传感器", purple),
        "logs": ((525, 835, 855, 980), "数据层\nlogs/agent_x\nmonitor.csv\nmodels / trajectories", blue),
        "outputs": ((80, 835, 410, 980), "输出层\noutputs/pygame_ui\nrewards/trajectory/comparison\noutputs/exports", green),
    }
    for xy, text, fill in boxes.values():
        box(draw, xy, text, fill, border, body)

    arrow(draw, (410, 222), (525, 222))
    arrow(draw, (855, 222), (970, 222))
    arrow(draw, (690, 295), (690, 390))
    arrow(draw, (855, 462), (970, 462))
    arrow(draw, (1148, 295), (1148, 390))
    arrow(draw, (1148, 535), (1148, 615))
    arrow(draw, (1148, 760), (1148, 835))
    arrow(draw, (970, 907), (855, 907))
    arrow(draw, (525, 907), (410, 907))
    arrow(draw, (245, 535), (245, 835))
    arrow(draw, (245, 390), (245, 295))

    draw.text(
        (90, 1010),
        "说明：run.py 负责训练/测试；Dashboard 和 monitor_plot.py 读取 logs 产物并生成 PNG 图像或导出归档。",
        font=font(16),
        fill=(54, 67, 82),
    )
    out = OUT_DIR / "software_architecture_diagram.png"
    img.save(out)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dev = make_development_flow()
    arch = make_architecture()
    print(dev)
    print(arch)


if __name__ == "__main__":
    main()

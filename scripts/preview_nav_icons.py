"""Preview navigation glyphs without loading or changing AC-Prof settings.

Run: uv run --locked python scripts/preview_nav_icons.py
List all pages: uv run --locked python scripts/preview_nav_icons.py --list
List image tree symbols: uv run --locked python scripts/preview_nav_icons.py --tree
Reference: Textual's docs/examples/widgets/option_list_options.py (MIT).
Optional glyph names: https://github.com/ryanoasis/nerd-fonts/blob/master/glyphnames.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from rich.cells import cell_len
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Button, Footer, OptionList, Static
from textual.widgets.option_list import Option

from acprof.tui.ui.navigation import PageNavigation
from acprof.tui.ui.presentation import COLLAPSED_SYMBOL, EXPANDED_SYMBOL
from acprof.tui.ui.rendering import CjkScreen
from acprof.tui.ui.themes import THEME_CATALOG, UI_THEMES


@dataclass(frozen=True)
class IconChoice:
    glyph: str
    name: str
    group: str = "线条与几何"
    note: str = "普通 Unicode；请在实际使用的终端里比较字形和对齐。"


CURRENT = PageNavigation.ICONS["run-tab"]
EMOJI_NOTE = "Emoji 的颜色、宽度和字形随终端字体变化，请以本机显示为准。"
CHOICES = (
    IconChoice(CURRENT, "当前图案"),
    IconChoice("▤", "横线面板 · 推荐"),
    IconChoice("☷", "参数清单 · 推荐"),
    IconChoice("⊞", "四格面板 · 推荐"),
    IconChoice("≣", "四条横线"),
    IconChoice("☰", "三条横线"),
    IconChoice("⌗", "井字网格"),
    IconChoice("▦", "矩阵网格"),
    IconChoice("▥", "竖线面板", note="当前也用于“实验结果”，可比较同图案带来的辨识差异。"),
    IconChoice("▧", "斜线面板"),
    IconChoice("▣", "方框中心"),
    IconChoice("□", "空心方框"),
    IconChoice("■", "实心方块"),
    IconChoice("◇", "空心菱形"),
    IconChoice("◆", "实心菱形"),
    IconChoice("◈", "菱形中心"),
    IconChoice("◎", "同心圆"),
    IconChoice("◉", "圆形中心"),
    IconChoice("◌", "虚线圆"),
    IconChoice("⌘", "交错环"),
    IconChoice("⊕", "圆框加号"),
    IconChoice("⊟", "方框横线"),
    IconChoice("⋮", "竖向三点"),
    IconChoice("⋯", "横向三点"),
    IconChoice("›", "单箭头"),
    IconChoice("»", "双箭头"),
    IconChoice("⚙", "齿轮", note="当前“设置”已用此图案，实验配置共用时辨识度较低。"),
    IconChoice("⚒", "工具"),
    IconChoice("⚗", "蒸馏瓶"),
    IconChoice("🧪", "试管", "Emoji", EMOJI_NOTE),
    IconChoice("🔬", "显微镜", "Emoji", EMOJI_NOTE),
    IconChoice("🧬", "双螺旋", "Emoji", EMOJI_NOTE),
    IconChoice("🎛️", "控制旋钮", "Emoji", EMOJI_NOTE),
    IconChoice("[=]", "ASCII 面板", "ASCII"),
    IconChoice("+", "ASCII 加号", "ASCII"),
    IconChoice("", "只显示文字", "无图案", "展开时只显示标题；折叠后只剩选中标记。"),
)

# Keep the existing interactive experiment choices and their numbers unchanged.
PAGE_CHOICES = {
    "run": ("实验配置", CHOICES),
    "monitor": ("运行监控", (
        IconChoice("◷", "时钟右上"),
        IconChoice("◴", "时钟左上"),
        IconChoice("◵", "时钟左下"),
        IconChoice("◶", "时钟右下"),
        IconChoice("◔", "四分之一圆"),
        IconChoice("◕", "四分之三圆"),
        IconChoice("◎", "同心圆"),
        IconChoice("◉", "圆形中心"),
        IconChoice("◌", "虚线圆"),
        IconChoice("⌁", "波形"),
        IconChoice("∿", "正弦波"),
        IconChoice("≋", "多条波形"),
        IconChoice("▶", "运行箭头"),
        IconChoice("⏱", "秒表", "Emoji"),
        IconChoice("📈", "上升曲线", "Emoji"),
        IconChoice("📡", "信号监控", "Emoji"),
        IconChoice(">", "ASCII 箭头", "ASCII"),
    )),
    "results": ("实验结果", (
        IconChoice("▤", "横线面板"),
        IconChoice("▦", "矩阵网格"),
        IconChoice("▧", "斜线面板"),
        IconChoice("▨", "反向斜线"),
        IconChoice("▩", "交叉网格"),
        IconChoice("▥", "竖线面板"),
        IconChoice("⊞", "四格面板"),
        IconChoice("≣", "四条横线"),
        IconChoice("≡", "三条横线"),
        IconChoice("∑", "汇总"),
        IconChoice("✓", "完成勾号"),
        IconChoice("📊", "柱状图", "Emoji"),
        IconChoice("📈", "折线图", "Emoji"),
        IconChoice("📋", "结果清单", "Emoji"),
        IconChoice("#", "ASCII 网格", "ASCII"),
    )),
    "backfill": ("补采实验", (
        IconChoice("↺", "逆向环箭头"),
        IconChoice("⟳", "顺向大环"),
        IconChoice("⟲", "逆向大环"),
        IconChoice("⤾", "顺向弧箭头"),
        IconChoice("⤿", "逆向弧箭头"),
        IconChoice("⇄", "往返箭头"),
        IconChoice("⇆", "交换箭头"),
        IconChoice("⊕", "圆框加号"),
        IconChoice("⊞", "方框加号"),
        IconChoice("↶", "回转箭头"),
        IconChoice("↷", "前向回转"),
        IconChoice("🔄", "循环双箭头", "Emoji"),
        IconChoice("🔁", "重复采集", "Emoji"),
        IconChoice("+", "ASCII 补充", "ASCII"),
    )),
    "images": ("镜像管理", (
        IconChoice("□", "空心方框"),
        IconChoice("▢", "圆角方框"),
        IconChoice("▣", "方框中心"),
        IconChoice("▦", "多格面板"),
        IconChoice("⬡", "空心六边形"),
        IconChoice("⬢", "实心六边形"),
        IconChoice("◇", "空心菱形"),
        IconChoice("◈", "菱形中心"),
        IconChoice("▱", "平面图层"),
        IconChoice("▰", "实心图层"),
        IconChoice("▧", "纹理面板"),
        IconChoice("📦", "镜像包", "Emoji"),
        IconChoice("🗃️", "存储盒", "Emoji"),
        IconChoice("🐳", "鲸鱼", "Emoji"),
        IconChoice("[]", "ASCII 方框", "ASCII"),
    )),
    "settings": ("设置", (
        IconChoice("⚙", "齿轮默认样式"),
        IconChoice("⚙︎", "齿轮文本样式"),
        IconChoice("⚒", "交叉工具"),
        IconChoice("⚒︎", "工具文本样式"),
        IconChoice("⚐", "标记旗"),
        IconChoice("☷", "参数清单"),
        IconChoice("☰", "菜单横线"),
        IconChoice("≡", "三条横线"),
        IconChoice("⊙", "调节旋钮"),
        IconChoice("⊛", "圆框星号"),
        IconChoice("⌘", "控制符号"),
        IconChoice("⋮", "更多选项"),
        IconChoice("⋯", "横向选项"),
        IconChoice("🎛️", "控制旋钮", "Emoji"),
        IconChoice("🔧", "扳手", "Emoji"),
        IconChoice("🛠️", "工具组", "Emoji"),
        IconChoice("*", "ASCII 星号", "ASCII"),
    )),
}

# Codicons names/codepoints checked against Nerd Fonts' glyphnames.json.
# Font assets are not bundled; rendering requires a client-side Nerd Font.
NERD_FONT_CHOICES = {
    "run": (("\ueb52", "cod-settings"), ("\uea79", "cod-beaker"),
            ("\ueb17", "cod-list_unordered"), ("\ueab3", "cod-checklist")),
    "monitor": (("\ueacd", "cod-dashboard"), ("\ueb31", "cod-pulse"),
                ("\ueb03", "cod-graph"), ("\uea70", "cod-eye")),
    "results": (("\uebb7", "cod-table"), ("\uebe2", "cod-graph_line"),
                ("\uebe3", "cod-graph_scatter"), ("\ueb9d", "cod-output")),
    "backfill": (("\ueb37", "cod-refresh"), ("\uea82", "cod-history"),
                 ("\uea77", "cod-sync"), ("\uead2", "cod-debug_restart")),
    "images": (("\ueb29", "cod-package"), ("\ueb50", "cod-server"),
               ("\ueace", "cod-database"), ("\uebd2", "cod-layers")),
    "settings": (("\ueaf8", "cod-gear"), ("\ueb6d", "cod-tools"),
                 ("\ueb51", "cod-settings_gear"), ("\ueb52", "cod-settings")),
}

TREE_ARROWS = {
    "A": ("当前箭头", COLLAPSED_SYMBOL.strip(), EXPANDED_SYMBOL.strip()),
    "B": ("小实心三角", "▸", "▾"),
    "C": ("空心三角", "▷", "▽"),
    "D": ("小空心三角", "▹", "▿"),
    "E": ("细角标", "›", "⌄"),
    "F": ("方框加减", "⊞", "⊟"),
    "G": ("简单加减", "+", "−"),
}
TREE_MARKERS = {
    "1": ("方框勾选", "□", "☑", "◩"),
    "2": ("复选框", "☐", "☑", "⊟"),
    "3": ("方块填充", "□", "■", "▣"),
    "4": ("圆形填充", "○", "●", "◐"),
}


def heading(glyph: str, title: str) -> str:
    return f"{glyph}  {title}" if glyph else title


def list_choices(console: Console, page: str | None, nerd_font: bool) -> None:
    console.print("AC-Prof · 六类导航图案候选", style="bold cyan")
    console.print("按页面分组；码位可区分相似图案。编号仅用于本清单。")
    console.print("普通 Unicode / Emoji 的字形、占宽取决于终端字体，请以实际显示为准。")
    if nerd_font:
        console.print("Nerd Font 项需要显示终端使用该字体；SSH 请配置客户端字体。")
    else:
        console.print("加 --nerd-font 可同时列出 Nerd Font / Codicons 候选。")
    for key, (title, candidates) in PAGE_CHOICES.items():
        if page is not None and key != page:
            continue
        current = PageNavigation.ICONS[f"{key}-tab"]
        choices = [IconChoice(current, "当前图案"), *candidates, CHOICES[-1]]
        if nerd_font:
            choices.extend(IconChoice(glyph, name, "Nerd Font")
                           for glyph, name in NERD_FONT_CHOICES[key])
        unique: dict[str, IconChoice] = {}
        for choice in choices:
            unique.setdefault(choice.glyph, choice)
        choices = list(unique.values())
        caption = f"{key} · {title} · {len(choices)} 项"
        compact = console.width < 80
        table = Table("编号", "效果", "名称", "类型", "码位",
                      title=caption, title_justify="left")
        console.print()
        if compact:
            console.print(caption, style="bold")
        for index, choice in enumerate(choices, 1):
            kind = "Unicode" if choice.group == "线条与几何" else choice.group
            codepoints = " ".join(f"U+{ord(char):04X}" for char in choice.glyph) or "—"
            sample = heading(choice.glyph, title)
            style = "cyan" if index == 1 else None
            if compact:
                console.print(Text(f"{index:02d}  {sample}\n    {choice.name}\n"
                                   f"    {kind} · {codepoints}", overflow="fold"), style=style)
            else:
                table.add_row(str(index), Text(sample), Text(choice.name), kind, codepoints,
                              style=style)
        if not compact:
            console.print(table)


def list_tree_choices(console: Console, arrow: str | None, marker: str | None) -> None:
    console.print("AC-Prof · 镜像树图案预览", style="bold cyan")
    console.print("固定示例展示未选、已选和部分选中；请在使用 TUI 的终端中比较字形。")
    console.print("箭头（折叠 / 展开）", style="bold")
    for key, (name, closed, opened) in TREE_ARROWS.items():
        console.print(Text(f"  {key} · {name}：{closed} / {opened}"))
    console.print("选择标记（未选中 / 已选中 / 部分选中）", style="bold")
    for key, (name, empty, checked, partial) in TREE_MARKERS.items():
        console.print(Text(f"  {key} · {name}：{empty} / {checked} / {partial}"))
    console.print("占用项的不可选标记保持为 —。")
    console.print()
    console.print("组合示例", style="bold")
    for arrow_id, (arrow_name, closed, opened) in TREE_ARROWS.items():
        if arrow is not None and arrow_id != arrow:
            continue
        for marker_id, (marker_name, empty, checked, partial) in TREE_MARKERS.items():
            if marker is not None and marker_id != marker:
                continue
            console.print(Text(f"{arrow_id}{marker_id} · {arrow_name} + {marker_name}"),
                          style="bold cyan")
            tree = Tree("", hide_root=True, guide_style="dim")
            tree.add(Text(f"{closed} {empty}  PyTorch CPU"))
            tree.add(Text(f"{closed} {checked}  PyTorch CUDA 12.8"))
            parent = tree.add(Text(f"{opened} {partial}  Python CPU"))
            parent.add(Text(f"{closed} {empty}  onnxruntime-cpu"))
            console.print(tree)
            console.print()
    console.print("只看某组合：--tree --arrow B --marker 2")


class IconPreviewApp(App[IconChoice | None]):
    TITLE = "AC-Prof 图案预览"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [
        Binding("r", "reset", "当前图案"),
        Binding("l", "language", "中/英标题"),
        Binding("t", "theme", "换配色"),
        Binding("ctrl+s", "choose", "选定"),
        Binding("escape,q", "quit", "退出"),
    ]
    CSS = """
    Screen { background: $background; color: $text; }
    #title { height: 2; padding: 1 2 0 2; color: $accent; text-style: bold; }
    #help { height: auto; padding: 0 2 1 2; color: $text-muted; }
    #body { height: 1fr; padding: 0 1; }
    #choices { width: 32; height: 1fr; border: round $panel; }
    #preview { width: 1fr; padding: 0 2; }
    .caption { height: auto; color: $text-muted; }
    #candidate-caption { margin-top: 1; }
    .sample-row { height: 3; }
    .sample {
        width: 1fr; height: 3; padding: 0 1;
        border-left: solid $accent; background: $boost;
        color: $accent; content-align: left middle; text-style: bold;
    }
    .folded { width: 5; margin-left: 2; padding: 0; content-align: center middle; }
    #details { height: auto; margin-top: 1; }
    #note { height: auto; color: $text-muted; }
    #actions { height: 3; margin: 0 2 1 2; align-horizontal: right; }
    #choose { min-width: 16; }
    Footer { background: $surface; }
    """

    def __init__(self, theme: str = "acprof-graphite") -> None:
        super().__init__()
        for palette in THEME_CATALOG:
            self.register_theme(Theme(**palette.theme_kwargs()))
        self.theme = theme
        self.choice = CHOICES[1]
        self.english = False

    def get_default_screen(self) -> Screen:
        return CjkScreen()

    def compose(self) -> ComposeResult:
        yield Static("AC-Prof · 导航图案预览", id="title")
        yield Static(f"上下键或点击切换 · 共 {len(CHOICES)} 种选择", id="help")
        with Horizontal(id="body"):
            options = []
            for index, choice in enumerate(CHOICES):
                glyph = Text(choice.glyph or "—", style="bold")
                glyph.pad_right(max(0, 3 - glyph.cell_len))
                prompt = Text(f"{index + 1:02d}  ") + glyph + Text(f" {choice.name}")
                options.append(Option(prompt, id=str(index)))
            yield OptionList(*options, id="choices")
            with VerticalScroll(id="preview"):
                yield Static("当前图案 / 右侧为折叠", classes="caption")
                with Horizontal(classes="sample-row"):
                    yield Static(id="original", classes="sample", markup=False)
                    yield Static(CURRENT, classes="sample folded", markup=False)
                yield Static("候选预览", id="candidate-caption", classes="caption")
                with Horizontal(classes="sample-row"):
                    yield Static(id="candidate", classes="sample", markup=False)
                    yield Static(id="folded", classes="sample folded", markup=False)
                yield Static(id="details", markup=False)
                yield Static(id="note", markup=False)
        with Vertical(id="actions"):
            yield Button("选定并退出", id="choose", variant="primary")
        yield Footer()

    def on_mount(self) -> None:
        choices = self.query_one("#choices", OptionList)
        choices.border_title = "选择图案"
        choices.highlighted = 1
        choices.focus()
        self.refresh_preview()

    def refresh_preview(self) -> None:
        title = "Experiment" if self.english else "实验配置"
        self.query_one("#original", Static).update(heading(CURRENT, title))
        self.query_one("#candidate", Static).update(heading(self.choice.glyph, title))
        self.query_one("#folded", Static).update(self.choice.glyph)
        codepoints = " ".join(f"U+{ord(char):04X}" for char in self.choice.glyph) or "无"
        self.query_one("#details", Static).update(
            f"{self.choice.name} · {self.choice.group}\n"
            f"{codepoints} · 渲染宽度 {cell_len(self.choice.glyph)} 格"
        )
        self.query_one("#note", Static).update(self.choice.note)

    @on(OptionList.OptionHighlighted, "#choices")
    def preview_choice(self, event: OptionList.OptionHighlighted) -> None:
        self.choice = CHOICES[event.option_index]
        self.refresh_preview()

    def action_reset(self) -> None:
        self.query_one("#choices", OptionList).highlighted = 0

    def action_language(self) -> None:
        self.english = not self.english
        self.refresh_preview()

    def action_theme(self) -> None:
        self.theme = UI_THEMES[(UI_THEMES.index(self.theme) + 1) % len(UI_THEMES)]

    @on(Button.Pressed, "#choose")
    def action_choose(self) -> None:
        self.exit(self.choice)


def main() -> None:
    parser = argparse.ArgumentParser(description="列出导航或镜像树图案；无参数时交互预览实验配置图案。")
    parser.add_argument("--list", action="store_true", help="按六类导航打印全部候选，不进入交互界面")
    parser.add_argument("--tree", action="store_true", help="打印镜像树的 7 组箭头、4 组标记及组合")
    parser.add_argument("--arrow", type=str.upper, choices=TREE_ARROWS,
                        help="只预览指定箭头的组合（自动启用镜像树列表）")
    parser.add_argument("--marker", choices=TREE_MARKERS,
                        help="只预览指定选择标记的组合（自动启用镜像树列表）")
    parser.add_argument("--page", choices=PAGE_CHOICES, help="只列出指定页面的候选（自动启用列表）")
    parser.add_argument("--nerd-font", action="store_true", help="列表中加入需要 Nerd Font 的图案")
    parser.add_argument("--theme", choices=UI_THEMES, default="acprof-graphite", help="预览配色")
    args = parser.parse_args()
    if args.tree or args.arrow is not None or args.marker is not None:
        if args.page is not None or args.nerd_font:
            parser.error("镜像树选项不能与 --page 或 --nerd-font 同用")
        list_tree_choices(Console(), args.arrow, args.marker)
        return
    if args.list or args.page is not None or args.nerd_font:
        list_choices(Console(), args.page, args.nerd_font)
        return
    selected = IconPreviewApp(theme=args.theme).run()
    if selected is not None:
        print(f"已选择：{selected.glyph or '（无图案）'}  {selected.name}")
        print(heading(selected.glyph, "实验配置"))


if __name__ == "__main__":
    main()

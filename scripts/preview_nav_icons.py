"""Preview navigation glyphs without loading or changing AC-Prof settings.

Run: uv run --locked python scripts/preview_nav_icons.py
Reference: Textual's docs/examples/widgets/option_list_options.py (MIT).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from rich.cells import cell_len
from rich.console import Console
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.theme import Theme
from textual.widgets import Button, Footer, OptionList, Static
from textual.widgets.option_list import Option

from acprof.tui.ui.navigation import PageNavigation
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
    IconChoice("▣", "方框中心", note="当前也用于“镜像管理”，可比较同图案带来的辨识差异。"),
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


def heading(glyph: str, title: str) -> str:
    return f"{glyph}  {title}" if glyph else title


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
    parser = argparse.ArgumentParser(description="预览实验配置的导航图案；选择只输出到终端。")
    parser.add_argument("--list", action="store_true", help="打印全部图案，不进入交互界面")
    parser.add_argument("--theme", choices=UI_THEMES, default="acprof-graphite", help="预览配色")
    args = parser.parse_args()
    if args.list:
        table = Table("编号", "图案", "名称", "效果", title="实验配置 · 图案候选")
        for index, choice in enumerate(CHOICES, 1):
            table.add_row(str(index), Text(choice.glyph or "—"), choice.name,
                          Text(heading(choice.glyph, "实验配置")))
        Console().print(table)
        return
    selected = IconPreviewApp(theme=args.theme).run()
    if selected is not None:
        print(f"已选择：{selected.glyph or '（无图案）'}  {selected.name}")
        print(heading(selected.glyph, "实验配置"))


if __name__ == "__main__":
    main()

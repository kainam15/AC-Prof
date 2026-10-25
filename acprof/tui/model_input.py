"""The model field remains free text, with local evidence reachable on demand."""
from textual.binding import Binding

from acprof.tui.app_contracts import candidate_picker_app
from acprof.tui.input import BarCursorInput


class ModelInput(BarCursorInput):
    BINDINGS = [Binding('f4', 'candidates', '模型候选', show=False)]

    def action_candidates(self):
        candidate_picker_app(self.app).models.open_model_candidates()

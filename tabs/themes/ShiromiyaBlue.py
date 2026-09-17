from gradio.themes.base import Base
from gradio.themes.utils import colors


class ShiromiyaBlue(Base):
    """Dark blue theme used by Shiromiya RVC Fork."""

    def __init__(self):
        super().__init__(
            primary_hue=colors.blue,
            secondary_hue=colors.blue,
            neutral_hue=colors.neutral,
        )
        self.name = ("ShiromiyaBlue",)
        super().set(
            body_background_fill="#061226",
            body_background_fill_dark="#061226",
            body_text_color="#e6f1ff",
            body_text_color_dark="#e6f1ff",
            body_text_color_subdued="#9db6d1",
            body_text_color_subdued_dark="#9db6d1",
            background_fill_primary="#081a33",
            background_fill_primary_dark="#081a33",
            background_fill_secondary="#0c2342",
            background_fill_secondary_dark="#0c2342",
            block_background_fill="#0d2748",
            block_background_fill_dark="#0d2748",
            block_border_color="#1d4f7a",
            block_border_color_dark="#1d4f7a",
            border_color_primary="#1d4f7a",
            border_color_primary_dark="#1d4f7a",
            input_background_fill="#07182d",
            input_background_fill_dark="#07182d",
            input_border_color="#2b638e",
            input_border_color_dark="#2b638e",
            button_primary_background_fill="#1565c0",
            button_primary_background_fill_dark="#1565c0",
            button_primary_background_fill_hover="#1e88e5",
            button_primary_background_fill_hover_dark="#1e88e5",
            button_primary_text_color="white",
            button_primary_text_color_dark="white",
            button_secondary_background_fill="#12365a",
            button_secondary_background_fill_dark="#12365a",
            button_secondary_background_fill_hover="#185086",
            button_secondary_background_fill_hover_dark="#185086",
            button_secondary_text_color="white",
            button_secondary_text_color_dark="white",
            color_accent="#42a5f5",
            color_accent_soft="#12365a",
            color_accent_soft_dark="#12365a",
            link_text_color="#64b5f6",
            link_text_color_dark="#64b5f6",
            link_text_color_hover="#90caf9",
            link_text_color_hover_dark="#90caf9",
            panel_background_fill="#0a1f3a",
            panel_background_fill_dark="#0a1f3a",
        )

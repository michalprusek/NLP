"""GTR Prompt Encoder - Reused from robust_vec2text.

Uses sentence-transformers/gtr-t5-base for Vec2Text compatibility.
No modifications needed - the GTR encoder is the same across both implementations.
"""

from robust_vec2text.encoder import GTRPromptEncoder

__all__ = ["GTRPromptEncoder"]

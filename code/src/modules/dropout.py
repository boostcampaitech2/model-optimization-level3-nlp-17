"""Fire module, generator.

- Author: kyungwon oh
- Contact: kyungwon.dev@gmail.com
"""
from torch import nn as nn

from src.modules.base_generator import GeneratorAbstract


class Dropout(nn.Dropout):
    """Dropout module."""

    def __init__(self, *args, **kwargs):  # pylint: disable=unused-argument
        super().__init__(*args, **kwargs)


class DropoutGenerator(GeneratorAbstract):
    """ Dropout module generator."""

    def __init__(self, *args, **kwargs):
        """Initailize."""
        super().__init__(*args, **kwargs)

    @property
    def out_channel(self) -> int:
        """Get out channel size."""
        return self.in_channel

    def __call__(self, repeat: int = 1):
        return self._get_module(Dropout(self.args[0]))

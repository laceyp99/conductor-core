"""Internal structured payload shared by variation provider adapters."""

from pydantic import BaseModel, ConfigDict

from conductor_core.models import Loop


class VariationCollection(BaseModel):
    """The private wire shape requested from every provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    variations: list[Loop]

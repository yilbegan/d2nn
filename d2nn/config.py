from pathlib import Path
from typing import Annotated

import msgspec
from msgspec import Meta

__all__ = [
    "Config",
    "ConfigError",
    "NonNegativeFloat",
    "NonNegativeInt",
    "PositiveFloat",
    "PositiveInt",
    "parse_document",
    "read_document",
]


class ConfigError(ValueError): ...


class Config(msgspec.Struct, frozen=True, forbid_unknown_fields=True): ...


type PositiveInt = Annotated[int, Meta(gt=0)]
type NonNegativeInt = Annotated[int, Meta(ge=0)]
type PositiveFloat = Annotated[float, Meta(gt=0.0)]
type NonNegativeFloat = Annotated[float, Meta(ge=0.0)]


def parse_document[T: Config](source: str, config: type[T], /) -> T:
    try:
        return msgspec.yaml.decode(source, type=config)
    except msgspec.ValidationError as error:
        raise ConfigError(str(error)) from error
    except msgspec.DecodeError as error:
        raise ConfigError(f"invalid YAML: {error}") from error


def read_document[T: Config](path: str | Path, config: type[T], /) -> T:
    file = Path(path)
    if file.suffix.lower() != ".yaml":
        raise ConfigError(f"configuration file must use the .yaml suffix: {path}")
    return parse_document(file.read_text(encoding="utf-8"), config)

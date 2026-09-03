import pathlib
from dataclasses import asdict, dataclass, replace
from typing import NotRequired, Self, TypedDict, cast, override

import torch
import torch.nn as nn
import torch.nn.functional as F

from d2nn.optics.parameters import PhysicalParameters
from d2nn.optics.stack import DiffractiveStack, DiffractiveStackConditions


@dataclass(frozen=True, slots=True)
class EncoderConfig:
    in_size: int = 32
    out_size: int = 200
    num_layers: int = 3
    num_classes: int = 10
    class_embedding_size: int = 32

    def __post_init__(self) -> None:
        if self.in_size < 1:
            raise ValueError("in_size must be at least 1")
        if self.out_size < 2:
            raise ValueError("out_size must be at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if self.class_embedding_size < 1:
            raise ValueError("class_embedding_size must be at least 1")


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    size: int = 200
    num_layers: int = 3
    wavelength: float = 5.32e-7
    refractive_index: float = 1.5
    extinction_coefficient: float = 0.0
    environment_refractive_index: float = 1.0
    pixel_size: float = 3.6e-5
    distance: float = 0.1
    base_thickness: float = 0.0
    scale_factor: float = 6.0
    quantization_levels: int | None = None

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2")
        if self.num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if self.quantization_levels is not None and self.quantization_levels < 2:
            raise ValueError("quantization_levels must be at least 2")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")
        if self.scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        if self.distance < 0:
            raise ValueError("distance must be non-negative")
        if self.base_thickness < 0:
            raise ValueError("base_thickness must be non-negative")
        if self.extinction_coefficient < 0:
            raise ValueError("extinction_coefficient must be non-negative")
        if self.refractive_index == self.environment_refractive_index:
            raise ValueError(
                "refractive_index must differ from environment_refractive_index"
            )

    @property
    def physical_parameters(self) -> PhysicalParameters:
        return PhysicalParameters(
            wavelength=self.wavelength,
            refractive_index=self.refractive_index,
            extinction_coefficient=self.extinction_coefficient,
            environment_refractive_index=self.environment_refractive_index,
        )


class _EncoderConfigData(TypedDict):
    in_size: int
    out_size: int
    num_layers: int
    num_classes: int
    class_embedding_size: int


class _DecoderConfigData(TypedDict):
    size: int
    num_layers: int
    wavelength: float
    refractive_index: NotRequired[float]
    extinction_coefficient: NotRequired[float]
    environment_refractive_index: NotRequired[float]
    pixel_size: float
    distance: float
    base_thickness: NotRequired[float]
    scale_factor: float
    quantization_levels: int | None


class _GenerativeCheckpoint(TypedDict):
    state_dict: dict[str, torch.Tensor]
    encoder_config: _EncoderConfigData
    decoder_config: _DecoderConfigData


class DiffractiveEncoder(nn.Module):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config: EncoderConfig = config or EncoderConfig()
        c = self.config

        feature_size = c.in_size**2 + c.class_embedding_size
        self.class_embedding: nn.Embedding = nn.Embedding(
            c.num_classes, c.class_embedding_size
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            nn.Linear(feature_size, feature_size) for _ in range(c.num_layers - 1)
        )
        self.activation: nn.LeakyReLU = nn.LeakyReLU(negative_slope=0.2)
        self.output_head: nn.Linear = nn.Linear(feature_size, c.in_size**2)
        self.amplitude_scaling_head: nn.Linear = nn.Linear(feature_size, 1)

    @override
    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = cast(torch.Tensor, self.class_embedding(labels))
        encoded = torch.cat((noise.flatten(1), embeddings), dim=1)

        for module in self.layers:
            layer = cast(nn.Linear, module)
            encoded = cast(torch.Tensor, self.activation(layer(encoded)))

        scale = cast(torch.Tensor, self.amplitude_scaling_head(encoded)).squeeze(-1)

        c = self.config
        phase = torch.tanh(cast(torch.Tensor, self.output_head(encoded)))
        phase = torch.pi * (phase.view(-1, c.in_size, c.in_size) + 1)
        phase = F.interpolate(
            phase.unsqueeze(1), size=c.out_size, mode="nearest"
        ).squeeze(1)
        field = torch.exp(1j * phase)

        return field, scale


class DiffractiveDecoder(nn.Module):
    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config: DecoderConfig = config or DecoderConfig()
        c = self.config

        self.stack: DiffractiveStack = DiffractiveStack(
            size=c.size,
            pixel_size=c.pixel_size,
            distance=c.distance,
            physical_parameters=c.physical_parameters,
            num_layers=c.num_layers,
            base_thickness=c.base_thickness,
            scale_factor=c.scale_factor,
            quantization_levels=c.quantization_levels,
        )
        self.conditions: DiffractiveStackConditions | None = None

    def set_conditions(self, conditions: DiffractiveStackConditions | None) -> None:
        self.conditions = conditions
        self.stack.set_conditions(conditions)

    @override
    def forward(self, field: torch.Tensor) -> torch.Tensor:
        field = cast(torch.Tensor, self.stack(field))
        intensity = field.abs().square()
        return intensity / intensity.amax(dim=(-2, -1), keepdim=True)


class DiffractiveGenerativeModel(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig | None = None,
        decoder_config: DecoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder: DiffractiveEncoder = DiffractiveEncoder(encoder_config)
        self.decoder: DiffractiveDecoder = DiffractiveDecoder(decoder_config)
        self.conditions: DiffractiveStackConditions | None = None

    def set_conditions(self, conditions: DiffractiveStackConditions | None) -> None:
        self.conditions = conditions
        self.decoder.set_conditions(conditions)

    @override
    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, scale = cast(
            tuple[torch.Tensor, torch.Tensor], self.encoder(noise, labels)
        )
        decoded = cast(torch.Tensor, self.decoder(encoded))
        return decoded, scale

    def save(self, path: str | pathlib.Path) -> None:
        checkpoint: _GenerativeCheckpoint = {
            "state_dict": self.state_dict(),
            "encoder_config": cast(
                _EncoderConfigData,
                cast(object, asdict(self.encoder.config)),
            ),
            "decoder_config": cast(
                _DecoderConfigData,
                cast(object, asdict(self.decoder.config)),
            ),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        device: torch.device | str = "cpu",
        *,
        quantization_levels: int | None = None,
    ) -> Self:
        checkpoint = cast(
            _GenerativeCheckpoint,
            torch.load(path, map_location=device, weights_only=True),
        )
        encoder_config = EncoderConfig(**checkpoint["encoder_config"])
        decoder_config = DecoderConfig(**checkpoint["decoder_config"])
        if quantization_levels is not None:
            decoder_config = replace(
                decoder_config, quantization_levels=quantization_levels
            )

        model = cls(encoder_config, decoder_config)
        _ = model.load_state_dict(checkpoint["state_dict"])
        return model

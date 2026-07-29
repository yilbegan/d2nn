import pathlib
from dataclasses import asdict, dataclass, replace

import torch
import torch.nn as nn

from d2nn.optics import DiffractiveLayer, Propagation


@dataclass(frozen=True)
class EncoderConfig:
    in_size: int = 32
    out_size: int = 200
    num_layers: int = 3
    num_classes: int = 10
    class_embedding_size: int = 32


@dataclass(frozen=True)
class DecoderConfig:
    size: int = 200
    num_layers: int = 3
    wavelength: float = 5.32e-7
    pixel_size: float = 3.6e-5
    distance: float = 0.1
    scale_factor: float = 6.0
    quantization_levels: int | None = None
    quantization_steepness: float = 4.0


class DiffractiveEncoder(nn.Module):
    def __init__(self, config: EncoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or EncoderConfig()
        c = self.config

        dim = c.in_size**2 + c.class_embedding_size
        self.class_embedding = nn.Embedding(c.num_classes, c.class_embedding_size)
        self.layers = nn.ModuleList(
            nn.Linear(dim, dim) for _ in range(c.num_layers - 1)
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.output_head = nn.Linear(dim, c.in_size**2)
        self.amplitude_scaling_head = nn.Linear(dim, 1)

    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.class_embedding(labels)
        features = torch.cat([noise.flatten(1), embeddings], dim=1)

        encoded = features
        for layer in self.layers:
            encoded = self.activation(layer(encoded))

        scale = self.amplitude_scaling_head(encoded).squeeze(-1)

        c = self.config
        encoded = torch.tanh(self.output_head(encoded))
        encoded = torch.pi * (encoded.view(-1, c.in_size, c.in_size) + 1)
        encoded = nn.functional.interpolate(
            encoded.unsqueeze(1), size=c.out_size, mode="nearest"
        ).squeeze(1)
        encoded = torch.exp(1j * encoded)

        return encoded, scale


class DiffractiveDecoder(nn.Module):
    def __init__(self, config: DecoderConfig | None = None) -> None:
        super().__init__()
        self.config = config or DecoderConfig()
        c = self.config

        self.layers = nn.ModuleList(
            DiffractiveLayer(
                c.size,
                c.pixel_size,
                c.wavelength,
                c.distance,
                c.scale_factor,
                c.quantization_levels,
                c.quantization_steepness,
            )
            for _ in range(c.num_layers)
        )

        self.output_propagation = Propagation(
            c.size, c.pixel_size, c.wavelength, c.distance
        )

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            field = layer(field)
        field = self.output_propagation(field)
        intensity = field.abs() ** 2
        return intensity / intensity.amax(dim=(-2, -1), keepdim=True)


class DiffractiveGenerativeModel(nn.Module):
    def __init__(
        self,
        encoder_config: EncoderConfig | None = None,
        decoder_config: DecoderConfig | None = None,
    ) -> None:
        super().__init__()
        self.encoder = DiffractiveEncoder(encoder_config)
        self.decoder = DiffractiveDecoder(decoder_config)

    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, scale = self.encoder(noise, labels)
        decoded = self.decoder(encoded)
        return decoded, scale

    def save(self, path: str | pathlib.Path) -> None:
        checkpoint = {
            "state_dict": self.state_dict(),
            "encoder_config": asdict(self.encoder.config),
            "decoder_config": asdict(self.decoder.config),
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        device: torch.device = torch.device("cpu"),
        *,
        quantization_levels: int | None = None,
        quantization_steepness: float | None = None,
    ):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        encoder_config = EncoderConfig(**checkpoint["encoder_config"])
        decoder_config = DecoderConfig(**checkpoint["decoder_config"])
        if quantization_levels is not None:
            decoder_config = replace(
                decoder_config, quantization_levels=quantization_levels
            )
        if quantization_steepness is not None:
            decoder_config = replace(
                decoder_config, quantization_steepness=quantization_steepness
            )
        model = cls(encoder_config, decoder_config)
        model.load_state_dict(checkpoint["state_dict"])
        return model

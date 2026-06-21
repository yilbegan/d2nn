import torch
import torch.nn as nn

from d2nn.optics import DiffractiveLayer, Propagation


class DiffractiveEncoder(nn.Module):
    def __init__(
        self,
        in_size: int = 32,
        out_size: int = 200,
        num_layers: int = 3,
        num_classes: int = 10,
        class_embedding_size: int = 32,
    ) -> None:
        super().__init__()
        dim = in_size**2 + class_embedding_size
        self.in_size = in_size
        self.out_size = out_size
        self.class_embedding = nn.Embedding(num_classes, class_embedding_size)
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(num_layers - 1))
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.output_head = nn.Linear(dim, in_size**2)
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

        encoded = torch.tanh(self.output_head(encoded))
        encoded = torch.pi * (encoded.view(-1, self.in_size, self.in_size) + 1)
        encoded = nn.functional.interpolate(
            encoded.unsqueeze(1), size=self.out_size, mode="nearest"
        ).squeeze(1)
        encoded = torch.exp(1j * encoded)

        return encoded, scale


class DiffractiveDecoder(nn.Module):
    def __init__(
        self,
        size: int = 200,
        num_layers: int = 3,
        wavelength: float = 5.32e-7,
        pixel_size: float = 3.6e-5,
        distance: float = 0.1,
        scale_factor: float = 6.0,
    ) -> None:
        super().__init__()
        self.size = size
        self.layers = nn.ModuleList(
            DiffractiveLayer(size, pixel_size, wavelength, distance, scale_factor)
            for _ in range(num_layers)
        )
        self.output_propagation = Propagation(size, pixel_size, wavelength, distance)

    def forward(self, field: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            field = layer(field)
        field = self.output_propagation(field)
        intensity = field.abs() ** 2
        return intensity / intensity.amax(dim=(-2, -1), keepdim=True)


class DiffractiveGenerativeModel(nn.Module):
    def __init__(
        self, encoder: DiffractiveEncoder, decoder: DiffractiveDecoder
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, noise: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, scale = self.encoder(noise, labels)
        decoded = self.decoder(encoded)
        return decoded, scale

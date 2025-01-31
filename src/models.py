import torch
import torch.nn as nn
import torchvision


def get_effnet(out: int = 1, freeze: bool = False) -> torch.nn.Module:
    """
    Perform some changes in architecture and returns model, arch (name)
    :param out: number of output neurons
    :param freeze: freeze conv layers
    :return: model
    """
    weights = torchvision.models.EfficientNet_B4_Weights.DEFAULT
    model = torchvision.models.efficientnet_b4(weights=weights)
    model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=0.2, inplace=True),
                torch.nn.Linear(1792, out),
            )
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
    return model


def get_inception(out: int = 1, freeze: bool = False) -> torch.nn.Module:
    """
    Perform some changes in architecture and returns model, arch (name)
    :param out: number of output neurons
    :param freeze: freeze conv layers
    :return: model
    """
    model = torchvision.models.inception_v3()
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(2048, out)

    return model


def get_vgg(out: int = 1, freeze: bool = False) -> torch.nn.Module:
    """
    Perform some changes in architecture and returns model, arch (name)
    :param out: number of output neurons
    :param freeze: freeze conv layers
    :return: model
    """
    model = torchvision.models.vgg16(torchvision.models.VGG16_Weights.IMAGENET1K_V1)
    if freeze:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[6] = torch.nn.Linear(4096, out)

    return model


def get_resnet(out: int = 1, freeze: bool = False) -> torch.nn.Module:
    """
    Perform some changes in architecture and returns model, arch (name)
    :param out: number of output neurons
    :param freeze: freeze conv layers
    :return: model
    """
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = torch.nn.Linear(2048, out)

    return model


def get_densenet(out: int = 1, freeze: bool = False) -> torch.nn.Module:
    """
    Perform some changes in architecture and returns model, arch (name)
    :param out: number of output neurons
    :param freeze: freeze conv layers
    :return: model
    """
    model = torchvision.models.densenet121(weights='IMAGENET1K_V1')
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.classifier = torch.nn.Linear(1024, out)

    return model


class Autoencoder(nn.Module):
  def __init__(self, encoder, decoder):
    super().__init__()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    self.encoder = encoder
    self.encoder.to(device)

    self.decoder = decoder
    self.decoder.to(device)

  def forward(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


class Decoder_v1(nn.Module):
  def __init__(self, bottleneck):
    super().__init__()

    self.fcl = torch.nn.Linear(bottleneck, 147852)
    self.decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(3, 3, 5, 1, 1, bias=True),
        torch.nn.ReLU(),
    )

  def forward(self, x):
      x = self.fcl(x)
      x = torch.nn.functional.relu(x)
      x = x.reshape((-1, 3, 222, 222))
      x = self.decoder(x)
      return x


class Autoencoder_v1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 6, 5, 1, 1, bias=True),
            torch.nn.ReLU(),
        )
        self.fcf = torch.nn.Linear(147852, 1000)
        self.fc1 = torch.nn.Linear(1000, 100)
        self.fcl = torch.nn.Linear(100, 147852)
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(6, 3, 5, 1, 1, bias=True),
            torch.nn.ReLU(),
        )

    def forward(self, x):

        latent = self.encoder(x)
        latent = latent.reshape((-1, 147852))

        latent = self.fcf(latent)
        latent = torch.nn.functional.relu(latent)
        latent = self.fc1(latent)

        latent = torch.nn.functional.relu(latent)
        latent = self.fcl(latent)
        latent = latent.reshape((-1, 6, 222, 222))
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed

        return reconstructed


if __name__ == "__main__":
    pass

import torch

from vggface_pretraining.models import TripletNetwork


def load_triplet_model(weights_path, model_name="resnet18", embedding_dim=128, pretrained=True, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TripletNetwork(model_name=model_name, embedding_dim=embedding_dim, pretrained=pretrained)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
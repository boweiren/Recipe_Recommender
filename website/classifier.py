import torch
from torchvision import datasets, transforms


device = torch.device('cpu')
model = torch.load(
    "./cached_models/ingredient_classifier_torch.pkl", map_location=device)

ingredient_dict = {
    0: "apple",
    1: "avocado",
    2: "banana",
    3: "beef",
    4: "bell pepper",
    5: "bread",
    6: "broccoli",
    7: "cabbage",
    8: "cheese",
    9: "chicken",
    10: "corn",
    11: "cucumber",
    12: "egg",
    13: "eggplant",
    14: "green bean",
    15: "lemon",
    16: "lettuce",
    17: "mushroom",
    18: "olive",
    19: "onion",
    20: "pasta",
    21: "potato",
    22: "rice",
    23: "salmon",
    24: "spinach",
    25: "tomato",
}


def predict_images(directory):
    xform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])
    inputs = datasets.ImageFolder(directory, transform=xform)
    loader = torch.utils.data.DataLoader(
        inputs, batch_size=len(inputs), shuffle=True)
    model.eval()

    with torch.no_grad():
        for samples, _ in loader:
            samples = samples.to(device)
            outs = model(samples)
            _, preds = torch.max(outs.detach(), 1)

    return preds.tolist()

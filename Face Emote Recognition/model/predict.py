from torchvision import transforms
from PIL import Image
import torch

def extract_features(image):
    img = Image.fromarray(image,"RGB")
    img = test_transform(img)
    img = img.unsqueeze(0)
    return img


labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'sad', 5 : 'suprise', 6 : 'neutral'}


test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.TenCrop(40),
        transforms.Lambda(lambda crops: torch.stack(
            [transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda(lambda tensors: torch.stack(
            [transforms.Normalize(mean=(0,), std=(255,))(t) for t in tensors])),
    ])

def prediction_model(image_batch,model):
    bs,nCrop,c,h,w = image_batch.shape
    image_batch = image_batch.view(-1,c,h,w)
    outputs = model(image_batch)
    outputs = outputs.view(bs, nCrop, -1)
    outputs = torch.sum(outputs, dim=1) / nCrop
    predicted_class = torch.argmax(outputs, dim = 1)

    return int(predicted_class[0])


def predict(image,model):
    image = extract_features(image)
    prediction = prediction_model(image,model)

    return labels[prediction]


def save_model(model, optimizer, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint["model"],)
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer




import torch
from torchvision import transforms
from PIL import Image
import os
from model import CoralCNN
from config import resize_x, resize_y, input_channels, num_classes
import glob


transform = transforms.Compose([
    transforms.Resize((resize_x, resize_y)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def classify_corals(list_of_img_paths):

    model = CoralCNN()
    model.load_state_dict(torch.load("checkpoints/final_weights.pth", map_location=torch.device('cpu')))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    images = []
    for path in list_of_img_paths:
        img = Image.open(path).convert("RGB")
        img = transform(img)
        images.append(img)

    batch = torch.stack(images).to(device)

    with torch.no_grad():
        outputs = model(batch)
        predicted_classes = torch.argmax(outputs, dim=1)

    labels = ["healthy" if pred == 1 else "bleached" for pred in predicted_classes]

    return labels


if __name__ == "__main__":

    coral_images = glob.glob("test_data/*.jpg") 
    
    if not coral_images:
        print("Error: No images found in 'data/' directory!")
        print("Please add coral images (JPG/PNG) to the 'data' folder.")
    else:
       
        predictions = classify_corals(coral_images)
        
        print("\nCoral Classification Results:")
        print("---------------------------")
        for img_path, label in zip(coral_images, predictions):
            print(f"{os.path.basename(img_path):<20} → {label}")
        print("\nDone!")


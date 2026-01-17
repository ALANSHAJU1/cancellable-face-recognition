import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

# -------------------------------
# Dataset Class
# -------------------------------
class CelebADataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = sorted([
            f for f in os.listdir(root_dir)
            if f.lower().endswith(".jpg")
        ])

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, img_path


# -------------------------------
# Test Loader
# -------------------------------
if __name__ == "__main__":
    dataset_path = "../datasets/celeba/img_align_celeba/img_align_celeba"

    dataset = CelebADataset(dataset_path)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Total images:", len(dataset))

    for i, (img, path) in enumerate(loader):
        print("Image tensor shape:", img.shape)
        print("Image path:", path[0])
        break

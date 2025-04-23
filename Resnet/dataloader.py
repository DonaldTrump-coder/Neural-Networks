from datasets import load_dataset
from datasets import Image
from torchvision import transforms
import torch
from torch.utils.data import Dataset

#dataset = load_dataset("parquet", data_files="data/valid-00000-of-00001-70d52db3c749a935.parquet")
#dataset["train"]=dataset["train"].cast_column("image", Image())

class MyDataset(Dataset):
    def __init__(self,input_dataset,transform=None):
        super().__init__()
        data = load_dataset("parquet", data_files=input_dataset)
        data["train"]=data["train"].cast_column("image", Image())
        self.dataset=data['train']
        self.transform=transform

    def __len__(self):
        return self.dataset.num_rows
    
    def __getitem__(self, index):
        item=self.dataset[index]
        image=item['image']
        label=item['label']

        # 强制转换为 RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image=self.transform(image)

        return image,label
    
    def get_num_classes(self):
        return len(set(self.dataset["label"]))
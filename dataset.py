import os

from torch.utils.data import Dataset
from PIL import Image

class Market1501Dataset(Dataset):
    def __init__(self, path, data='train', transform=None, load_img=True):
        self.path = path
        self.data = data
        self.load_img = load_img
        self.transform = transform

        if data == 'train':
            data_path = os.path.join(path, 'bounding_box_train')
        elif data == 'test':
            data_path = os.path.join(path, 'bounding_box_test')
        elif data == 'query':
            data_path = os.path.join(path, 'query')
        else:
            raise Exception('Invalid dataset type')

        self.samples = []
        images = os.listdir(data_path)

        for img in images:
            if not img.endswith('jpg'):
                continue

            img_path = os.path.join(data_path, img)
            label, cam_seq, frame, bbox = img.split('_')           

            # Camera, Sequence, and BBox ID
            cam = int(cam_seq[1])
            seq = int(cam_seq[3])
            bbox = int(bbox.split('.')[0])

            self.samples.append((img_path, int(label), cam, seq, int(frame), int(bbox)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label, cam, _, _, _ = self.samples[index]

        if self.data == 'train':
            label = index

        if self.load_img:
            img = Image.open(img_path)
            if self.transform is not None:
                img = self.transform(img)
            return label, img, cam

        return label, cam
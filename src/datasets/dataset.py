import glob
import os
import cv2
import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np

from .registry import dataset


def process_data(dataset_path):
    data = []
    filenames = []
    supported_extensions = ["bmp", "tif", "jpg", "png"]

    for ext in supported_extensions:
        data.extend(glob.glob(os.path.join(dataset_path, f"*.{ext}")))

    data.sort()
    filenames = [os.path.basename(file) for file in data]
    return data, filenames


class BaseDataset(Dataset):
    def __init__(self, dataset_split, data_dir_vis, data_dir_ir, data_dir_label=None):
        super(BaseDataset, self).__init__()

        assert dataset_split in ['train', 'test'], 'dataset_split must be "train" or "test"'

        self.dataset_split = dataset_split

        self.filepath_vis, self.filenames_vis = process_data(data_dir_vis)
        self.filepath_ir, self.filenames_ir = process_data(data_dir_ir)

        if data_dir_label is not None:
            self.filepath_label, self.filenames_label = process_data(data_dir_label)

        if len(self.filenames_vis) != len(self.filenames_ir):
            raise ValueError("Number of visual and infrared files must be equal.")

        self.length = len(self.filenames_vis)

    def __getitem__(self, index):
        # Get file paths
        vis_path = self._get_path(index, 'vis')
        ir_path = self._get_path(index, 'ir')

        # Load images
        image_vis = np.asarray(Image.open(vis_path))  # (h, w, c), (480, 640, 3)
        image_inf = cv2.imread(ir_path, 0)  # (480, 640)

        # Convert images to tensors
        image_vi = self._image_to_tensor(image_vis, is_vis=True)  # c, h, w
        image_ir = self._image_to_tensor(image_inf, is_vis=False)  # h, w
        image_ir = np.expand_dims(image_ir, axis=0)  # 1, h, w
        name = self.filenames_vis[index]

        if hasattr(self, 'filepath_label'):
            label_path = self._get_path(index, 'label')
            label = np.asarray(Image.open(label_path), dtype=np.int64)  # (480, 640)
            return image_vi, image_ir, label, name
        else:
            return image_vi, image_ir, name

    def __len__(self):
        return self.length

    def _get_path(self, index, image_kind):
        if image_kind == 'vis':
            return self.filepath_vis[index]
        elif image_kind == 'ir':
            return self.filepath_ir[index]
        elif image_kind == 'label':
            return self.filepath_label[index]

    def _image_to_tensor(self, image, is_vis):
        if is_vis:
            out = np.asarray(Image.fromarray(image), dtype=np.float32).transpose((2, 0, 1)) / 255.0
        else:
            out = np.asarray(Image.fromarray(image), dtype=np.float32) / 255.0

        return torch.tensor(out)


@dataset.register
class MSRSDataSet(BaseDataset):
    def __init__(self, dataset_split, data_dir_vis, data_dir_ir, data_dir_label):
        super(MSRSDataSet, self).__init__(dataset_split, data_dir_vis, data_dir_ir, data_dir_label)


@dataset.register
class LLVIPDataSet(BaseDataset):
    def __init__(self, dataset_split, data_dir_vis, data_dir_ir):
        super(LLVIPDataSet, self).__init__(dataset_split, data_dir_vis, data_dir_ir)


if __name__ == '__main__':
    # Import the required libraries
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    # Set the paths to your data directories
    data_dir_vis = r'..\..\data\MSRS\Visible\train\MSRS'
    data_dir_ir = r'..\..\data\MSRS\Infrared\train\MSRS'
    data_dir_label = r'..\..\data\MSRS\Label\train\MSRS'
    dataset_split = 'train'

    # Create an instance of the MSRSDataSet for the test split
    data_set = MSRSDataSet(dataset_split, data_dir_vis, data_dir_ir, data_dir_label)

    # Create a data loader to iterate over the dataset
    dataloader = DataLoader(data_set, batch_size=1, shuffle=True)

    # Iterate over the dataset and visualize a few samples
    for images_vi, images_ir, labels, names in dataloader:
        # Access the individual images and labels
        image_vi = images_vi.squeeze(0)  # Remove batch dimension
        image_ir = images_ir.squeeze(0)
        label = labels.squeeze(0)
        name = names[0]

        # Convert tensors to numpy arrays and transpose dimensions if needed
        image_vi = image_vi.numpy().transpose((1, 2, 0))
        image_ir = np.squeeze(image_ir.numpy())
        label = label.numpy()

        # Display the images and label
        plt.figure()
        plt.subplot(131)
        plt.imshow(image_vi)
        plt.title('Visual Image')
        plt.subplot(132)
        plt.imshow(image_ir, cmap='gray')
        plt.title('Infrared Image')
        plt.subplot(133)
        plt.imshow(label, cmap='gray')
        plt.title('Label')
        plt.show()

        # Print the name and label for reference
        print('Name:', name)
        print('Label:', label)

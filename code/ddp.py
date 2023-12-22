import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
import cv2
import time
import pandas as pd
import dask.array as da
import dask
from dask.diagnostics import ProgressBar
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp


# Model definition
class ConvolutionalModel(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.net = nn.Sequential(
            ## ConvBlock 1
            nn.Conv2d(3, 6, kernel_size=4, stride=1, padding=0),
            # Input: (b, 3, 256, 256) || Output: (b, 6, 250, 250)
            nn.BatchNorm2d(6),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=5, stride=5, padding=0),
            # Input: (b, 6, 250, 250) || Output: (b, 6, 50, 50)

            ## ConvBlock 2
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            # Input: (b, 6, 50, 50) || Output: (b, 16, 46, 46)
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # Input: (b, 16, 46, 46) || Output: (b, 16, 23, 23)

            ## ConvBlock 3
            nn.Conv2d(16, 32, kernel_size=8, stride=1, padding=0),
            # Input: (b, 16, 23, 23) || Output: (b, 32, 16, 16)
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=4, stride=4, padding=0),
            # Input: (b, 32, 16, 16) || Output: (b, 32, 4, 4)

            ## ConvBlock 4
            nn.Conv2d(32, 120, kernel_size=4, stride=1, padding=0),
            # Input: (b, 32, 4, 4) || Output: (b, 120, 1, 1)
            nn.BatchNorm2d(120),
            nn.Tanh(),
            nn.Flatten(),  # flat to a vector
            # Input: (b, 120, 1, 1) || Output: (b, 120*1*1) = (b, 120)

            nn.Dropout(p=0.32), # Avoid Overfitting
            ## DenseBlock
            nn.Linear(120, 84),
            # Input: (b, 120) || Output: (b, 84)
            nn.Tanh(),
            nn.Linear(84, output_size)
            # Input: (b, 84) || Output: (b, 10)
        )

    def forward(self, X):
        output = self.net(X)
        return output

# Custom dataset class
class ImagesLabelsDataset(Dataset):
    def __init__(self, images_array, labels_array):
        self.labels = labels_array
        self.images = images_array

    def __getitem__(self, idx):
        sample = self.images[idx]
        label = self.labels[idx]
        return sample, label

    def __len__(self):
        return len(self.labels)
    
# Define data processing functions and dataset class
def process_image(img):
    try:
        pic = cv2.imread(img)
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic = cv2.resize(pic, (256, 256), interpolation=cv2.INTER_AREA)
        pic = np.array(pic, dtype=np.float32)
        return pic
    except:
        return np.zeros(shape=(256, 256, 3), dtype=np.float32)


def main(rank, world_size): 
# Your data reading and processing
    data = []
    num_epochs = 20  # Set your desired number of epochs
    batch_size = 64  # Set your desired batch size
    for dirname, _, filenames in os.walk('./dataset'):
        for filename in filenames:
            label = dirname.split("/")[-1]
            image = os.path.join(dirname, filename)
            data.append((label, image))

    df = pd.DataFrame(data, columns=["label", "image"])

    unique_labels = df['label'].unique()
    unique_labels

    # for label in unique_labels:
    #     examples = pd.concat([examples, df.query(f"label == '{label}'").sample(1)])

    list_of_paths = df["image"].to_numpy()

    # images = [process_image(path) for path in list_of_paths]
    # images = np.array(images, dtype=np.float32)
    # print(images.shape)
    process_image_delayed = dask.delayed(process_image)
    images = [process_image_delayed(path) for path in list_of_paths]

    # Compute the results in parallel
    with ProgressBar():
        images = dask.compute(*images)

    # Convert the list to a Dask array
    images = da.from_array(np.array(images, dtype=np.float32))
    images = images.compute()
    print(images.shape)

    unique_labels = list(unique_labels) # Use for indexing the labels - ("hail", 0), ("rainbow", 1)

    labels = df["label"].to_numpy()

    # Using the unique labels list
    def convert_categorical_to_number(label):
        return unique_labels.index(label)

    func = np.vectorize(convert_categorical_to_number)
    labels = func(labels)
    labels


    # Split data into train and test sets
    images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2)

    images_train = torch.from_numpy(images_train).permute(0, 3, 1, 2)
    images_test = torch.from_numpy(images_test).permute(0, 3, 1, 2)
    labels_train = torch.from_numpy(labels_train)
    labels_test = torch.from_numpy(labels_test)

    # DDP Initialization
    world_size = torch.cuda.device_count()
    # print("Number of available GPUs:",world_size)
    

    # Initialize dataset and dataloaders
    train_set = ImagesLabelsDataset(images_train, labels_train)
    test_set = ImagesLabelsDataset(images_test, labels_test)
    
    train_sampler = DistributedSampler(train_set, num_replicas = world_size, rank = rank)

    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler = train_sampler, pin_memory = True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Initialize the model, optimizer, criterion...
    device = torch.device("cuda", rank) if torch.cuda.is_available() else torch.device("cpu")
    output_size = len(unique_labels)

    net = ConvolutionalModel(output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    net = nn.parallel.DistributedDataParallel(net, device_ids=[rank])
    # Training loop
    
    s = time.time()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        start_time = time.time()

        epoch_loss = []
        pred_list, label_list = np.array([]), np.array([])

        for i, (image, label) in enumerate (train_loader):
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            pred_label = net(image)
            loss = criterion(pred_label, label)
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.cpu().data)
            _, pred = torch.max(pred_label, axis=1)
            pred_list = np.append(pred_list, pred.cpu().numpy())
            label_list = np.append(label_list, label.cpu().numpy())

        epoch_loss = np.asarray(epoch_loss)
        epoch_acc = accuracy_score(label_list, pred_list)
        end_time = time.time()

        print("Rank: %d || Epoch: %d || Loss: %.3f || Accuracy:  %.2f || Time: %.2f" % (rank, epoch, epoch_loss.mean(), epoch_acc, end_time - start_time))
    e = time.time()
    print("Number of available GPUs:",world_size)
    print(e-s)


    # End of training
    torch.distributed.destroy_process_group()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size,), nprocs=world_size)

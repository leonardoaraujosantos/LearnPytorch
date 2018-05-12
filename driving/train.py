import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
from drive_dataset import DriveData
from torch.utils.data import DataLoader
from model import CNNDriver

# Library that gives support for tensorboard and pytorch
from tensorboardX import SummaryWriter

import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper Parameters
num_epochs = 100
batch_size = 4000
learning_rate = 0.01
L2NormConst = 0.001

# Tensorboard writer at logs directory
writer = SummaryWriter('logs')
cnn = CNNDriver()
cnn.train()
print(cnn)
writer.add_graph(cnn, torch.rand(10,3,66,200))
# Put model on GPU
cnn = cnn.to(device)

transformations = transforms.Compose([
    transforms.ToTensor(), transforms.Normalize(mean = [ 0.5, 0.5, 0.5 ],std = [ 0.5, 0.5, 0.5 ])])

# Instantiate a dataset
dset_train = DriveData('./Track1_Wheel_Cam/', transformations)
dset_train.addFolder('./Track2_Wheel_Cam/')
dset_train.addFolder('./Track3_Wheel_Cam/')
dset_train.addFolder('./Track4_Wheel_Cam/')
dset_train.addFolder('./Track5_Wheel_Cam/')
dset_train.addFolder('./Track6_Wheel_Cam/')
dset_train.addFolder('./Track7_Wheel_Cam/')

train_loader = DataLoader(dset_train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1, # 1 for CUDA
                          pin_memory=True # CUDA only
                         )


# Loss and Optimizer
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate, weight_decay=L2NormConst)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

print('Train size:',len(dset_train), 'Batch size:', batch_size)
print('Batches per epoch:',len(dset_train) // batch_size)
# Train the Model
for epoch in range(num_epochs):
    iteration_count = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # Send inputs/labels to GPU
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = cnn(images)
        loss = loss_func(outputs, labels.unsqueeze(dim=1))

        loss.backward()
        optimizer.step()

        # Send loss to tensorboard
        writer.add_scalar('loss/', loss.item(), epoch)

        # Display on each epoch
        if batch_idx == 0:
            # Send image to tensorboard
            writer.add_image('Image', images[batch_idx], epoch)
            # Print Epoch and loss
            print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
            # Save the Trained Model parameters
            torch.save(cnn.state_dict(), 'cnn_' + str(epoch) + '.pkl')

        iteration_count += 1
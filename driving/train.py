import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision.transforms as transforms
from drive_dataset import DriveData
from drive_dataset import DriveData_LMDB
from drive_dataset import AugmentDrivingTransform
from drive_dataset import DrivingDataToTensor
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
batch_size = 400
learning_rate = 0.0001
L2NormConst = 0.001

# Tensorboard writer at logs directory
writer = SummaryWriter('logs')
cnn = CNNDriver()
cnn.train()
print(cnn)
writer.add_graph(cnn, torch.rand(10, 3, 66, 200))
# Put model on GPU
cnn = cnn.to(device)

transformations = transforms.Compose([
    AugmentDrivingTransform(), DrivingDataToTensor()])

# Instantiate a dataset
#dset_train = DriveData('./Track1_Wheel_Cam/', transformations)
dset_train = DriveData_LMDB('/home/leoara01/work/DLMatFramework/virtual/tensorDriver/Track_Joystick_MultiCam_LMDB_Balanced/', transformations)
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
#optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

print('Train size:',len(dset_train), 'Batch size:', batch_size)
print('Batches per epoch:',len(dset_train) // batch_size)
# Train the Model
iteration_count = 0
for epoch in range(num_epochs):
    for batch_idx, samples in enumerate(train_loader):

        # Send inputs/labels to GPU
        images = samples['image'].to(device)
        labels = samples['label'].to(device)

        optimizer.zero_grad()

        # Forward + Backward + Optimize
        outputs = cnn(images)
        loss = loss_func(outputs, labels.unsqueeze(dim=1))

        loss.backward()
        optimizer.step()
        exp_lr_scheduler.step(epoch)

        # Send loss to tensorboard
        writer.add_scalar('loss/', loss.item(), iteration_count)
        writer.add_histogram('steering_out', outputs.clone().detach().cpu().numpy(), iteration_count)
        writer.add_histogram('steering_in', labels.unsqueeze(dim=1).clone().detach().cpu().numpy(), iteration_count)

        # Get current learning rate (To display on Tensorboard)
        for param_group in optimizer.param_groups:
            curr_learning_rate = param_group['lr']
            writer.add_scalar('learning_rate/', curr_learning_rate, iteration_count)

        # Display on each epoch
        if batch_idx == 0:
            # Send image to tensorboard
            writer.add_image('Image', images[batch_idx], epoch)
            writer.add_text('Steering', 'Steering:' + str(outputs[batch_idx].item()), epoch)
            # Print Epoch and loss
            print('Epoch [%d/%d] Loss: %.4f' % (epoch + 1, num_epochs, loss.item()))
            # Save the Trained Model parameters
            torch.save(cnn.state_dict(), 'cnn_' + str(epoch) + '.pkl')

        iteration_count += 1
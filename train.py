import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
from torchvision import models
from dataset import Pcam
from test import test
import time
import click


def train(model, trainloader, criterion, optimizer, epoch, log_interval, device, writer):
    model.train()
    running_loss = 0
    for batch_idx, data in enumerate(trainloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int64)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # lr_scheduler.step()
        running_loss += loss.item()

        if batch_idx % log_interval == 0:
            writer.add_scalar('training loss', running_loss/log_interval, epoch*len(labels)+batch_idx)
            print('Train Epoch: {} [{}/{}] \t Loss:{:.6f}'.format(
              epoch, batch_idx*len(labels), len(trainloader.dataset), running_loss/log_interval))
            running_loss = 0


@click.command()
@click.option('--batch-size', type=int, default=64, help='batch size')
@click.option('--lr', type=float, default=0.002, help='learning rate')
@click.option('--seed', type=int, default=1, help='random seed')
@click.option('--save-model', default=True, help='flag for save')
@click.option('--log-interval', type=int, default=30, help='log status interval')
def main(batch_size, lr, seed, save_model, log_interval):
    # params setting
    params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 1}
    class_num = 2

    # tensorboard setting
    writer = SummaryWriter('runs/pcam_experiment_1')

    # random seed setting
    torch.manual_seed(seed)

    # device setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.current_device())
    print(device)

    # define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # load training set
    trainset = Pcam('data/camelyonpatch_level_2_split_train_x.h5', 'data/camelyonpatch_level_2_split_train_y.h5', transform)
    trainloader = data.DataLoader(trainset, **params)

    # load validation set
    validset = Pcam('data/camelyonpatch_level_2_split_valid_x.h5', 'data/camelyonpatch_level_2_split_valid_y.h5', transform)
    validloader = data.DataLoader(validset, **params)

    # load pretrained model
    model = models.resnet50(pretrained=True)

    # freeze op
    # for param in model.parameters():
    #     param.requires_grad = False

    fc_in_feature = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(fc_in_feature, 512),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(512, class_num),
        nn.Softmax(dim=1),
    )
    model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()

    # setting optimizer
    ignored_params = list(map(id, model.fc.parameters()))
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': 1e-3},
        {'params': model.fc.parameters(), 'lr': 2e-3}
    ], weight_decay=1e-4)
    # optimizer = optim.Adam(model.fc.parameters(), lr=lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    # lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=1, epochs=16)

    # clock setting
    start = time.time()

    # train + valid
    for epoch in range(10):
        print('learning rate: {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        train(model, trainloader, criterion, optimizer, epoch, log_interval, device, writer)
        # trainset.close()
        test(model, validloader, criterion, device, writer, epoch, True)
        # validset.close()
        lr_scheduler.step()
        if save_model:
            torch.save(model.state_dict(), './checkpoints/epoch{}.pth'.format(epoch))

    print('Finish Training, Total Time: {}'.format(time.time()-start))


if __name__ == "__main__":
    main()

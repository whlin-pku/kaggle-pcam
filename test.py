import torch
from torchvision import transforms
from torchvision import models
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from dataset import Pcam
import time
import click
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import albumentations
from albumentations import pytorch as AT


def test(model, testloader, criterion, device, writer, epoch, flag):
    model.eval()
    test_loss = 0
    correct = 0
    y_true = []
    y_score = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device, dtype=torch.int64)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            labels = labels.view_as(pred)
            correct += pred.eq(labels).sum().item()
            labels = labels.cpu().numpy()
            pred = pred.cpu().numpy()
            y_true = y_true + labels.tolist()
            y_score = y_score + pred.tolist()
    test_loss = test_loss / len(testloader)

    if flag:
        writer.add_scalar('valid/test acc', correct/len(testloader.dataset), epoch)

    print('Test Set: Average loss: {:.4f}, Accuracy: {}/{}, {:.2f}%'.format(
      test_loss, correct, len(testloader.dataset), 100.*correct/len(testloader.dataset)
    ))

    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    auc_roc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, '-', label='Area Under the Curve (AUC = %0.4f)' % auc_roc)
    plt.title('ROC curve')
    plt.xlabel("FPR (False Positive Rate)")
    plt.ylabel("TPR (True Positive Rate)")
    plt.legend(loc="lower right")
    plt.show()
    print('Test Set AUC: {:.4f}'.format(auc_roc))


@click.command()
@click.option('--model-state-path', default='./checkpoints/epoch6.pth', help='checkpoint path')
def main(model_state_path):
    # parameters setting
    params = {'batch_size': 64, 'shuffle': False, 'num_workers': 1}
    class_num = 2

    # tensorboard setting
    writer = SummaryWriter('runs/pcam_experiment_1')

    # device setting
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(torch.cuda.current_device())
    print(device)

    # define transform
    # transform = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Resize(224),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                          std=[0.229, 0.224, 0.225]),
    # ])
    transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        AT.ToTensor()
    ])

    # load test data
    testset = Pcam('data/camelyonpatch_level_2_split_test_x.h5', 'data/camelyonpatch_level_2_split_test_y.h5', transform)
    testloader = data.DataLoader(testset, **params)

    # load model
    model = models.resnet50(pretrained=False)
    fc_in_feature = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.6),
        nn.Linear(fc_in_feature, 512, bias=True),
        nn.SELU(),
        nn.Dropout(p=0.8),
        nn.Linear(512, class_num, bias=True),
    )
    model.load_state_dict(torch.load(model_state_path))
    model.to(device)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # clock setting
    start = time.time()

    # test
    test(model, testloader, criterion, device, writer, 0, False)
    # testset.close()

    print('Total Time: {}'.format(time.time()-start))


if __name__ == '__main__':
    main()
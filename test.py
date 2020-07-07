import torch
from torch.autograd import Variable
from torchvision import transforms
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from dataset import IMAGE_Dataset
import numpy as np
from utils import parse_args

args = parse_args()

CUDA_DEVICES = args.cuda_devices
DATASET_ROOT = './test'
PATH_TO_WEIGHTS = args.weight + '/model-1.00-best_train_acc.pth' # Your model name
# PATH_TO_WEIGHTS = args.weight + '/epoch_models/model-70.0epoch.pth' # Your model name

def test():
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),

    ])
    test_set = IMAGE_Dataset(Path(DATASET_ROOT), data_transform)
    data_loader = DataLoader(
        dataset=test_set, batch_size=32, shuffle=True, num_workers=1)
    classes = [_dir.name for _dir in Path(DATASET_ROOT).glob('*')]
    classes.sort()
    classes.sort(key = len)

    # Load model
    model = torch.load(PATH_TO_WEIGHTS)
    model = model.cuda(CUDA_DEVICES)
    model.eval()
    

    total_correct = 0
    total = 0
    class_correct = list(0. for i in enumerate(classes))
    class_total = list(0. for i in enumerate(classes))

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = Variable(inputs.cuda(CUDA_DEVICES))
            labels = Variable(labels.cuda(CUDA_DEVICES))
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            # totoal
            total += labels.size(0)
            total_correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            
            # batch size
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    print("class_total : ", class_total)
    for i, c in enumerate(classes):
        print('Accuracy of %5s : %8.4f %%' % (
        c, 100 * class_correct[i] / class_total[i]))

    # Accuracy
    print('\nAccuracy on the ALL test images: %.4f %%'
      % (100 * total_correct / total))



if __name__ == '__main__':
    test()




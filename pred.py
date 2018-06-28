import os
import numpy as np
import pandas as pd
from dataset.dataset import dataset, collate_fn
import torch
from torch.nn import CrossEntropyLoss
import torch.utils.data as torchdata
from torchvision import datasets, models, transforms
from torchvision.models import resnet50
from cnn_finetune import make_model
from torchvision.models import resnet18
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from math import ceil
from  torch.nn.functional import softmax
from models.multiscale_resnet import multiscale_resnet
test_transforms= transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mode ="test"

rawdata_root = './datasets/'
all_pd = pd.read_csv("./datasets/train.txt",sep=" ",
                       header=None, names=['ImageName', 'label'])
train_pd, val_pd = train_test_split(all_pd, test_size=0.15, random_state=43,
                                    stratify=all_pd['label'])
true_test_pb = pd.read_csv("./datasets/test.txt",sep=" ",
                       header=None, names=['ImageName'])
"addFakeLabel"
true_test_pb['label'] =1

test_pd =true_test_pb if mode=="test" else val_pd
print(test_pd.head())

data_set = {}
data_set['test'] = dataset(imgroot=os.path.join(rawdata_root, mode), anno_pd=test_pd,
                             transforms=test_transforms,
                             )
data_loader = {}
data_loader['test'] = torchdata.DataLoader(data_set['test'], batch_size=4, num_workers=4,
                                           shuffle=False, pin_memory=True, collate_fn=collate_fn)

model_name = 'inceptionv4_nomal_label'
resume = './model/inceptionv4/weights-18-378-[0.9193].pth'

# model =resnet18(pretrained=True)
# model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=1)
# model.fc = torch.nn.Linear(model.fc.in_features,100)
#model =multiscale_resnet(num_class=100)
model = make_model('inception_v4', num_classes=100, pretrained=True) #3


print('resuming finetune from %s'%resume)
model.load_state_dict(torch.load(resume))
model = model.cpu()
model.eval()
criterion = CrossEntropyLoss()

if not os.path.exists('./result'):
    os.makedirs('./result')

test_size = ceil(len(data_set['test']) / data_loader['test'].batch_size)
test_preds = np.zeros((len(data_set['test'])), dtype=np.float32)
true_label = np.zeros((len(data_set['test'])), dtype=np.int)
idx = 0
test_loss = 0
test_corrects = 0
if __name__ == '__main__':
    for batch_cnt_test, data_test in enumerate(data_loader['test']):
        # print data
        print("{0}/{1}".format(batch_cnt_test, int(test_size)))
        inputs, labels = data_test
        inputs = Variable(inputs.cpu())
        labels = Variable(torch.from_numpy(np.array(labels)).long().cpu())
        # forward
        outputs = model(inputs)

        # statistics
        if isinstance(outputs, list):
            loss = criterion(outputs[0], labels)
            loss += criterion(outputs[1], labels)
            outputs = (outputs[0]+outputs[1])/2
        else:
            loss = criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)

        test_loss += loss.item()
        batch_corrects = torch.sum((preds == labels)).item()
        test_corrects += batch_corrects
        test_preds[idx:(idx + labels.size(0))] = preds
        true_label[idx:(idx + labels.size(0))] = labels.data.cpu().numpy()
        #   statistics
        idx += labels.size(0)
    test_loss = test_loss / test_size
    test_acc = 1.0 * test_corrects / len(data_set['test'])
    print('test-loss: %.4f ||test-acc@1: %.4f'
          % (test_loss, test_acc))

    test_pred = test_pd[['ImageName']].copy()
    test_pred['label'] = list(test_preds)
    test_pred['label'] = test_pred['label'].apply(lambda x: int(x)+1)
    test_pred[['ImageName',"label"]].to_csv('result/csv/{0}_{1}.csv'.format(model_name,mode) ,sep=" ",
                                                                 header=None, index=False)
    print(test_pred.info())

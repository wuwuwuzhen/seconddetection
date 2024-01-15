import torch
from lanelines.rundef import args_setting
from lanelines.dataset import RoadSequenceDataset, RoadSequenceDatasetList
from lanelines.model import generate_model
from torchvision import transforms
from torch.optim import lr_scheduler
from PIL import Image
import numpy as np
import cv2
import time
from lanelines.rundef import rename,resize,unetlstmtxt,detection
import lanelines.rundef
import os


def output_result(args,model, test_loader, device):
    model.eval()
    k = 0
    feature_dic=[]
    with torch.no_grad():
        for sample_batched in test_loader:
            k+=1
            #print(k)
            data, target = sample_batched['data'].to(device), sample_batched['label'].type(torch.LongTensor).to(device)
            output,feature = model(data)
            feature_dic.append(feature)
            pred = output.max(1, keepdim=True)[1]
            img = torch.squeeze(pred).cpu().unsqueeze(2).expand(-1,-1,3).numpy()*255
            img = Image.fromarray(img.astype(np.uint8))

            data = torch.squeeze(data).cpu().numpy()
            if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
                data = np.transpose(data[-1], [1, 2, 0]) * 255
            else:
                data = np.transpose(data, [1, 2, 0]) * 255
            data = Image.fromarray(data.astype(np.uint8))
            rows = img.size[0]
            cols = img.size[1]
            for i in range(0, rows):
                for j in range(0, cols):
                    img2 = (img.getpixel((i, j)))
                    if (img2[0] > 200 or img2[1] > 200 or img2[2] > 200):
                        data.putpixel((i, j), (234, 53, 57, 255))
            data = data.convert("RGB")
            data.save(lanelines.rundef.save_path + "%s_data.jpg" % k)#red line on the original image
            img.save(lanelines.rundef.save_path + "%s_pred.jpg" % k)#prediction result


def get_parameters(model, layer_name):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.UpsamplingBilinear2d
    )
    for name, module in model.named_children():
        if name in layer_name:
            for layer in module.children():
                if isinstance(layer, modules_skipped):
                    continue
                else:
                    for parma in layer.parameters():
                        yield parma

def main(picture_path):
    start = time.process_time()
    resize(picture_path)
    unetlstmtxt()
    args = args_setting()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)

    # turn image into floatTensor
    op_tranforms = transforms.Compose([transforms.ToTensor()])
    # load data for batches, num_workers for multiprocess
    if args.model == 'SegNet-ConvLSTM' or 'UNet-ConvLSTM':
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDatasetList(file_path=lanelines.rundef.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size, shuffle=False, num_workers=1)
    else:
        test_loader = torch.utils.data.DataLoader(
            RoadSequenceDataset(file_path=lanelines.rundef.test_path, transforms=op_tranforms),
            batch_size=args.test_batch_size, shuffle=False, num_workers=1)

    # load model and weights
    model = generate_model(args)
    model.to(device)
    class_weight = torch.Tensor(lanelines.rundef.class_weight)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)

    pretrained_dict = torch.load(lanelines.rundef.pretrained_path,map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    pretrained_dict_1 = {k: v for k, v in pretrained_dict.items() if (k in model_dict)}
    model_dict.update(pretrained_dict_1)
    model.load_state_dict(model_dict)

    # output the result pictures
    output_result(args, model, test_loader, device)
    end3 = time.process_time()
    rename()
    return(detection())

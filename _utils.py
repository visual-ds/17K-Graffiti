import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from torchvision.utils import make_grid
from torchvision import transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

#GPU / CPU
def SET_DEVICE(GPU, device_idx = 0):
    if GPU:
        device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    return device


#transformers
def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(0.5),
        ToTensorV2(p=1.0)],
        bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

def get_test_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


# get the model
def get_model_FRCNN():
    # model definitions and configs
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    return model


def collate_fn(batch):
    return tuple(zip(*batch))

def plot_img_bbox(tensor, target,width=6):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)

    for bbox in target:
        draw.rectangle(bbox, width=width, outline='green')

    from numpy import array as to_numpy_array
    return torch.from_numpy(to_numpy_array(im))


def draw_bounding_box_with_scores(tensor, bboxes, scores, width=4, thresh=0.8):
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(im)

    true_bboxes = 0
    for bbox, src in zip(bboxes, scores):
        if src < thresh: continue
        true_bboxes += 1
        draw.rectangle(bbox, width=width, outline='red')

    from numpy import array as to_numpy_array
    return torch.from_numpy(to_numpy_array(im)), true_bboxes



#this class is used to construct our graffiti dataset
class GraffitiDataset(Dataset):
    def __init__(self, image_dir, imagedata,w,h, transforms = None, file_extenstion='.jpg'):
        super().__init__()
        self.image_dir = image_dir
        self.imagedata = imagedata
        self.transforms = transforms
        self.width=w
        self.height=h
        self.f_extension = file_extenstion

    def __getitem__(self, idx):
        image_name = self.imagedata.iloc[idx]['FileName']
        bboxes = np.array(self.imagedata.iloc[idx]['bbox'])
        try:
            img = Image.open(os.path.join(self.image_dir, image_name+self.f_extension)).convert('RGB')
        except:
            raise(f'Error: No image file exist with name {image_name}')
        w, h = img.size
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        img = np.asarray(img).astype('float32')/255.0

        boxes_resize=[]
        for bb in bboxes:
            xmin,ymin,xmax,ymax= bb
            xmin_corr = (xmin/w)*self.width
            if xmax > w:
                print(xmax_corr)
            xmax_corr = (xmax/w)*self.width
            ymin_corr = (ymin/h)*self.height
            ymax_corr = (ymax/h)*self.height

            boxes_resize.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

        boxes = np.array(boxes_resize,dtype='float32')
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        area = torch.as_tensor(area,dtype=torch.float32)
        iscrowd = torch.zeros((np.shape(boxes)[0],), dtype=torch.int64)
        labels = torch.ones((np.shape(boxes)[0],), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms is not None:
            sample = {
                'image': img,
                'bboxes': target['boxes'],
                'labels': labels
            }
            sample = self.transforms(**sample)
            img = sample['image']
            target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
            target['boxes'] = target['boxes'].float()

        return img.float(), target, image_id

    def __len__(self)->int:
        return len(self.imagedata)


def intersection_over_union(gt_box, pred_box):
    inter_box_top_left = [max(gt_box[0], pred_box[0]), max(gt_box[1], pred_box[1])]
    inter_box_bottom_right = [min(gt_box[0] + gt_box[2], pred_box[0] + pred_box[2]),
                              min(gt_box[1] + gt_box[3], pred_box[1] + pred_box[3])]

    inter_box_w = inter_box_bottom_right[0] - inter_box_top_left[0]
    inter_box_h = inter_box_bottom_right[1] - inter_box_top_left[1]

    intersection = inter_box_w * inter_box_h
    union = gt_box[2] * gt_box[3] + pred_box[2] * pred_box[3] - intersection

    iou = intersection / union

    return iou, intersection, union


def average_precision(prediction, targets, index_on_target, iou_threshold):
    TP = torch.zeros(len(prediction['boxes']))
    FP = torch.zeros(len(prediction['boxes']))
    for gt_idx in range(len(targets[index_on_target]['boxes'])):
        best_iou = 0.0
        for prd_idx in range(len(prediction['boxes'])):
            iou, intersection, union = intersection_over_union(
                targets[index_on_target]['boxes'][gt_idx].cpu().detach().numpy(),
                prediction['boxes'][prd_idx].cpu().detach().numpy())
            if iou >= best_iou:
                best_iou = iou

            if best_iou >= iou_threshold:
                TP[prd_idx] = 1.0
            else:
                FP[prd_idx] = 1.0

    epsilon = 1e-6
    TP_cumsum = torch.cumsum(TP, dim=0)
    FP_cumsum = torch.cumsum(FP, dim=0)
    recalls = TP_cumsum / (len(np.where(TP == 1)[0]) + epsilon)
    precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
    precisions = torch.cat((torch.tensor([1]), precisions))
    recalls = torch.cat((torch.tensor([0]), recalls))

    return torch.trapz(precisions, recalls)
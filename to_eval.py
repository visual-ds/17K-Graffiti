from _utils import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Evaluate Faster R-CNN on detecting graffiti.')
parser.add_argument('--testImages', required=True,
                    metavar="/path/to/graffiti/dataset/test/",
                    help='Path to testing set images.')

parser.add_argument('--testBboxs', required=True,
                    metavar="/path/to/graffiti/annotation/test/",
                    help='Path to testing set annotations file.')

parser.add_argument('--weights', required=True,
                    metavar="/path/to/graffiti/model/weights/file",
                    help='Path to Graffiti pre-trained weights file.')
parser.add_argument('--iou', required=True, default=0.50, help='Set the IOU that AP can meet.')

parser.add_argument('--gpu', required=False, default=True, help='True: if to use GPU, ow False')
parser.add_argument('--deviceIdx', required=False, default=0, help='Set the gpu device index. Default is 0.')
parser.add_argument('--imageSize', required=False, default=(224,224), help='Set the image size.')
parser.add_argument('--batchSize', required=False, default=16, help='Set the batch size for training.')
parser.add_argument('--lrn', required=False, default=0.0001, help='Set the learning rate for training.')

args = parser.parse_args()
PATH_2_TESTING_IMGS = args.testImages
PATH_2_TESTING_BBOXS = args.testBboxs
PATH_2_MODEL_WEIGHTS = args.weights
IOU_THRESHOLD = float(args.iou)

GPU = args.gpu
DEVICE_IDX = int(args.deviceIdx)
IMAGE_SIZE = args.imageSize
BATCH_SIZE = int(args.batchSize)

test_bboxes = pd.read_pickle(PATH_2_TESTING_BBOXS)
test_ds = GraffitiDataset(PATH_2_TESTING_IMGS,
                           test_bboxes,
                           w=IMAGE_SIZE[0],
                           h=IMAGE_SIZE[1],
                           transforms=get_train_transform())
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

device = SET_DEVICE(GPU, DEVICE_IDX)

model = get_model_FRCNN()
model.load_state_dict(torch.load(PATH_2_MODEL_WEIGHTS))

model.eval()
model.to(device)

epsilon = 1e-6
average_precisions=[]
for images, targets, image_ids in test_loader:
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    image_ids = list(image_id.to(device) for image_id in image_ids)
    predictions = model(images)
    for i, prediction in enumerate(predictions):
        average_precisions.append(average_precision(prediction, targets, i, IOU_THRESHOLD))
mAP = sum(average_precisions) / len(average_precisions)
print(f'mAP@[iou={IOU_THRESHOLD}] = {mAP}')
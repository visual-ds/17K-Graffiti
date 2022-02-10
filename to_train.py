from _utils import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Train Faster R-CNN to detect graffiti.')
parser.add_argument('--trainImages', required=True,
                    metavar="/path/to/graffiti/dataset/train/",
                    help='Path to training set images.')

parser.add_argument('--trainBboxs', required=True,
                    metavar="/path/to/graffiti/annotation/train/",
                    help='Path to training set annotations file.')

parser.add_argument('--gpu', required=False, default=True, help='True: if to use GPU, ow False')
parser.add_argument('--deviceIdx', required=False, default=0, help='Set the gpu device index. Default is 0.')
parser.add_argument('--imageSize', required=False, default=(224,224), help='Set the image size.')
parser.add_argument('--batchSize', required=False, default=16, help='Set the batch size for training.')
parser.add_argument('--lrn', required=False, default=0.0001, help='Set the learning rate for training.')


args = parser.parse_args()
PATH_2_TRAINING_IMGS = args.trainImages
PATH_2_TRAINING_BBOXS = args.trainBboxs
GPU = args.gpu
DEVICE_IDX = int(args.deviceIdx)
IMAGE_SIZE = args.imageSize
BATCH_SIZE = int(args.batchSize)
LEARNING_RATE = float(args.lrn)


device = SET_DEVICE(GPU, DEVICE_IDX)

train_bboxes = pd.read_pickle(PATH_2_TRAINING_BBOXS)

train_ds = GraffitiDataset(PATH_2_TRAINING_IMGS,
                           train_bboxes,
                           w=IMAGE_SIZE[0],
                           h=IMAGE_SIZE[1],
                           transforms=get_train_transform())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)


model = get_model_FRCNN()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)


model.train()
model.to(device)

itr = 1
loss_values = []
NUM_EPOCH = 30
for epoch in range(NUM_EPOCH):
    for images, targets, image_ids in train_loader:

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if itr % 50 == 0:
            print(f"Iteration #{itr} loss: {loss_value}")

        loss_values.append(loss_value)

        itr += 1

#save model weights and loss hist
torch.save(model.state_dict(), 'graffiti_fasterrcnn_resnet50_fpn.pth')
np.save('FRCNN_training_loss.npy',loss_values)
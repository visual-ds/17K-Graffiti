from _utils import *
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Validate the graffiti detector.')
parser.add_argument('--imageDir', required=True,
                    metavar="/path/to/image/directory/",
                    help='Path to set of images.')

parser.add_argument('--outDir', required=True,
                    metavar="/path/to/output/directory/",
                    help='Path to output directory.')

parser.add_argument('--weights', required=True,
                    metavar="/path/to/graffiti/model/weights/file",
                    help='Path to Graffiti pre-trained weights file.')

parser.add_argument('--confidence', required=False, default=0.50, help='Set the prediction confidence.')

parser.add_argument('--imgWidth', required=False, default=480, help='Set the image size.')
parser.add_argument('--imgHeight', required=False, default=480, help='Set the image size.')

parser.add_argument('--gpu', required=False, default=True, help='True: if to use GPU, ow False')
parser.add_argument('--deviceIdx', required=False, default=0, help='Set the gpu device index. Default is 0.')
args = parser.parse_args()


device = SET_DEVICE(args.gpu, args.deviceIdx)
model = get_model_FRCNN()
model.load_state_dict(torch.load(args.weights))
model.eval()
model.to(device)

transform = T.Compose([T.Scale((int(args.imgWidth),int(args.imgHeight))),T.ToTensor()])

image_file_list = os.listdir(args.imageDir)
f_names = []
bboxes =[]

for i, f in enumerate(image_file_list):
    if f.endswith(('.jpg','.JPG','.png'))==False: continue
    img = Image.open(args.imageDir+f)
    img_transf = transform(img).float().unsqueeze(0)
    prd = model(img_transf.to(device))
    bbx = prd[0]['boxes'].cpu().detach().numpy()
    res_img, bbox_count = draw_bounding_box_with_scores(img_transf[0],
                                    prd[0]['boxes'].cpu().detach().numpy(),
                                    prd[0]['scores'].cpu().detach().numpy(),
                                    thresh=float(args.confidence))

    img_final = Image.fromarray(res_img.cpu().detach().numpy())
    img_final.save(os.path.join(args.outDir,f))

    for b in bbx:
        f_names.append(f)
        bboxes.append(b)

pd.DataFrame({'FileName':f_names,'Bbox':bboxes}).to_csv(os.path.join(args.outDir,'detection_result.csv'),index=False, sep=';')

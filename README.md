# 17K-Graffiti
This repository provides the graffiti dataset (17K-Graffiti) with the accepted paper "17K-Graffiti: Spatial and Crime Data Assessments in São Paulo City" on the VISAPP-2022 conference. The ultimate goal is to seek relations between Graffiti Incidence (as an spatial city’s element), and Crime Occurrences (as social offences). To this aim, we utilized this developed Graffiti detector to accomplish the goal of our project. 

See a quick demo of our developed graffiti detector on the provided jupyter notebook at `demo.ipynb`. 

# Download Dataset and Pre-trained Weights:
The 17K-Graffiti dataset and its pre-trained weights are available at this [link](https://zenodo.org/record/5899631).

# Dataset Annotation
The dataset boundary box annotations are in the directory /dataset/. It provides the annotations of training and testing sets individualy. Once you downloaded the dataset as well as its pre-trained weights, you can further have a quick toturial through the provided jupyter notebook at:
`/dataset/dataset_processing.ipynb`

# Requirements
python 3.0 &
pytorch > 1.9

# To Train
In order to train the graffiti model from the scratch run the following code:
```
python to_train.py --train_image <Path to training set> 
                   --train_bboxs <Path to training bboxs> 
                   --gpu True 
                   --batchSize 16
```

# To Evaluate
Use the following commande line to evaluate the performance of the graffiti detection model on the provided IOU:

```
python to_eval.py --testImages <Path to testing set>
                  --testBboxs <Path to testing bboxes>
                  --weights <path to pretrained model weights file>
                  --gpu True
                  --iou <given iou>
                  --batchSize 16
```

# To Perform Detection
To perform the graffiti detection on the set of images, use the following command. It will output the detected bounday boxes on the images and store the bbox information on a CSV file within the specified output directory. One can also set the image dimension and the model confidence on detection.  

```
python to_detect.py --imageDir <Path to images> 
                    --outDir <Path to output directory>  
                    --weights <path to pretrained model weights file>
                    --imgWidth <image width> 
                    --imgHeight <image height> 
                    --confidence <prediction confidence>
```

# Corresponding bibtex:

```
@conference{visapp22,
author={Bahram Lavi. and Eric Tokuda. and Felipe Moreno{-}Vera. and Luis Nonato. and Claudio Silva. and Jorge Poco.},
title={17K-Graffiti: Spatial and Crime Data Assessments in São Paulo City},
booktitle={Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP,},
year={2022},
pages={968-975},
publisher={SciTePress},
organization={INSTICC},
doi={10.5220/0010883300003124},
isbn={978-989-758-555-5},
}
```

# Contact us  
For any issue please kindly email to `bahram [dot] lavi [at] fgv [dot] br`

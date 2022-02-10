# 17K-Graffiti
This repository provides the graffiti dataset (17K-Graffiti) with the accepted paper "17K-Graffiti: Spatial and Crime Data Assessments in São Paulo City" on the VISAPP-2022 conference. The ultimate goal is to seek relations between Graffiti Incidence (as an spatial city’s element), and Crime Occurrences (as social offences). To this aim, we utilized this developed Graffiti detector to accomplish the goal of our project. 

See a quick demo of our developed graffiti detector on the provided jupyter notebook at `demo.ipynb`. 

# Download Dataset and Pre-trained Weights:
The 17K-Graffiti dataset and its pre-trained weights are available at this [link](https://zenodo.org/record/5899631).

# Requirements
python 3.0 &
pytorch > 1.9

# Dataset Annotation
The dataset boundary box annotations are in the directory /dataset/. It provides the annotations of training and testing sets individualy. Once you downloaded the dataset, you can have also a quick toturial through the provided jupyter notebook at `/dataset/dataset_processing.ipynb`

# To Train
In order to train the graffiti model from the scratch run the following code:
```
python to_train.py --train_image <Path to training set> 
                   --train_bboxs <Path to training bboxs> 
                   --gpu=True 
                   --batchSize 16
```

# To Evaluate
Use the following commande line to evaluate the performance of the graffiti detection model on the provided IOU:

```
python to_eval.py --testImages <Path to test set>
                   --testBboxs <Path to test bboxes>
                   --weights <path to pretrained model weights file>
                   --gpu True
                   --iou <given iou>
                   --batchSize 16
```

# To Perform Detection
Will be released soon!

# Corresponding bibtex to this repository
Will be reported soon!

# Contact us  
For any issue please kindly email to `bahram [dot] lavi [at] fgv [dot] br`

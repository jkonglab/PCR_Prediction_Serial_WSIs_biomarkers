# SpatialAtten2

```
SpatialAtten
│   README.md
│   colorDeconv.py      ## color deconvolution for biomarker detection 
|   attenMap.py         ## based on tumor detection and biomarker detection to generate spatial attention
|   resnet.py           ## resnet description for PCR prediction
|   train.py            ## train the PCR prediction model
│
└───colorDeconv
│   │   {img_id}_ki_deconv.jpg     ## Ki-67 color deconv results
│   │   {img_id}_phh_deconv.jpg    ## PHH3 color deconv results
|
└───detectResults
│   │   {img_id}.npz    ## tumor detection results from Mask R-CNN
|
└───attenMap
│   │   {img_id}.npy    ## attention map generated for each image patch
|
└───train_image
│   └───pcr
│   |   │   {img_id}_HES.npy    ## all attention quipped HE images for PCR prediction from pcr cases
│   |   │   {img_id}_KI-67.npy  ## all attention quipped Ki-67 images for PCR prediction from pcr cases
│   |   │   {img_id}_PHH3.npy   ## all attention quipped PHH3 images for PCR prediction from pcr cases
│   └───non-pcr
│   |   │   {img_id}_HES.npy    ## all attention quipped HE images for PCR prediction from pcr cases
│   |   │   {img_id}_KI-67.npy  ## all attention quipped Ki-67 images for PCR prediction from pcr cases
│   |   │   {img_id}_PHH3.npy   ## all attention quipped PHH3 images for PCR prediction from pcr cases

```

First run Mask R-CNN for tumor detection and run colorDeconv.py for biomarker detection. The run attenMap.py to generate spatial attention maps. Finally, run train.py on spatial attention equipped images for PCR prediction.

resnet part is referred from https://github.com/raghakot/keras-resnet.git

# Reference

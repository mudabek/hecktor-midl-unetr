# Automatic Segmentation of Head and Neck Tumor: How Powerful Transformers Are?
### [[Paper]](https://arxiv.org/abs/2201.06251)

Code for the Medical Imaging with Deep Learning (MIDL) 2022 conference paper studying performance of three different models (nnU-Net, Squeeze-and-Excitation U-Net, and UNETR) on the task of head and neck tumor segmentation. 


### Architecture

<p align="center">
  <img src="assets/model.jpg" alt="UNETR Architecture" width="700"/>
</p>


### Main requirements
- PyTorch 1.11.0 (cuda 10.2)
- SimpleITK 2.1.1
- nibabel 3.2.2
- skimage 0.19.2


### Dataset
Train and test images are available through the competition [website](https://www.aicrowd.com/challenges/miccai-2021-hecktor).


### Training the models
```
model_trainers/train_transformer.py -p [path_to_config]
```
Note: nnU-Net code can be directly used from [here](https://github.com/MIC-DKFZ/nnUNet). We recommend following [this](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjBosWMvNb2AhVQa8AKHSpTDyMQFnoECAoQAQ&url=https%3A%2F%2Fmedium.com%2Fmiccai-educational-initiative%2Fnnu-net-the-no-new-unet-for-automatic-segmentation-8d655f3f6d2a&usg=AOvVaw3UAd0OhSVNG-KYHp_4PcC8) article for using nnU-Net.

### Validation and Testing Results
Validation set results obtained on the folds available in train_configs folder. Testing set results are the results on the 2021 HECKTOR competition.
<p align="center">
  <img src="assets/results.jpg" alt="Results" width="700"/>
</p>


### Qualitative Results
<p align="center">
  <img src="assets/output_example.jpg" alt="Examples of segmentations by UNETR" width="700"/>
</p>

### Model Weights
[nnU-Net](https://drive.google.com/file/d/1z99JBTfAcA0mvCBFL7lB94IZa-7NvM30/view?usp=sharing) <br>
[SE-U-Net](https://drive.google.com/file/d/1z99JBTfAcA0mvCBFL7lB94IZa-7NvM30/view?usp=sharing) <br>
[UNETR](https://drive.google.com/file/d/1z99JBTfAcA0mvCBFL7lB94IZa-7NvM30/view?usp=sharing) <br>


### Paper
If you use this code in you research, please cite the following paper ([arXiv](https://arxiv.org/abs/2201.06251)):
> Sobirov I., Nazarov O., Alasmawi H., Mohammad Yaqub (2022) Automatic Segmentation of Head and Neck Tumor: How Powerful Transformers Are?

### References
[Original implementation of Squeeze-and-Excitation U-Net](https://github.com/iantsen/hecktor)

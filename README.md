This repository is for a U-Net based segmentaion model for surgical instruments and anatomies for laparoscopic hysterectomy procedures. 
The dataset used is AutoLaparo:

`
    @InProceedings{wang2022autolaparo,
        title = {AutoLaparo: A New Dataset of Integrated Multi-tasks for Image-guided Surgical Automation in Laparoscopic Hysterectomy},
        author = {Wang, Ziyi and Lu, Bo and Long, Yonghao and Zhong, Fangxun and Cheung, Tak-Hong and Dou, Qi and Liu, Yunhui},
        booktitle = {International Conference on Medical Image Computing and Computer-Assisted Intervention},
        pages = {486--496},
        year = {2022},
        organization = {Springer}
    }
`

For this project, we employed the architecture as the foundation for our segmentation model inspired by the U-Net architecture [16]. We leveraged the Keras framework and its prebuilt functionalities for the design and construction process. The implementation of the model involved the integration of convolutional layers, ReLU activation functions, dropout layers and max pooling operations. 
 ![image](https://github.com/rashad-h/Surgical-Scene-Understanding-Laparoscopic/assets/61196340/12e9a707-314b-4a4b-a2fd-e0a18306496d)


The architecture, as illustrated in Figure above, was designed to capture intricate details within the input images and accurately identify surgical instruments and anatomy of the uterus. Commencing with the input images of dimensions 256 by 256 by 3, with 3 representing the standard RGB channels, the model goes through a series of transformational layers. The initial layer, performs input normalization by scaling the pixel values to a range between 0 and 1, hence facilitating a smoother convergence during the training process. Afterwards, a convolutions layer with 3 by 3 filters and a feature size of 16 is applied, resulting in feature maps of dimensions 256 by 256 by16.

Following this a dropout layer is introduced to solve the overfitting issue by randomly deactivating a fraction of neurons while training. This process iterates through additional convolutional layers and max pooling operations, reducing the spatial dimensions while increasing the depth of the feature maps. This feature extraction process, finally results in a compact representation of the input image, captured in feature maps of dimensions 16 by 16 by 256 at the lowest level.
To continue with the segmentation process, transpose convolutional layers are implemented to scale up the feature maps, gradually restoring the resolution while reducing the depth of the feature maps. This up sampling operation is complemented by concatenation with feature maps from earlier convolutional layers as shown in Figure 11, enabling the model to integrate both low-level and high-level features for more accurate segmentation. 

The iterative process of convolutional, dropout and transpose convolutional layers continues until the resolution of the prediction matches the original input resolution of 256 by 256. The final convolutional layer equipped with the ReLU activation function, produces segmentation prediction with feature dimensions aligned with the number of segmentation classes, in this case 10.

Given that the segmentation masks predicted are encoded in a one-hot format, the SoftMax activation function emerged as the logical solution for the final layer. By applying the SoftMax activation, we ensured that the model’s output probabilities were normalized across all classes, with each value indicating the pixel’s likelihood of belonging to a specific class. 

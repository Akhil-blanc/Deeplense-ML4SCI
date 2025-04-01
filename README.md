# Deeplense-ML4SCI
### Problem Statement
- Common task: Classification of images into strong lensing images with no substructure, subhalo substructure, and vortex substructure.
- Specific task VI.A: Pretrain Masked AutoEncoder on the no_sub samples to learn a feature representation of strong lensing images.(Reconstruction of masked portions) 
---
### Approach
#### Common Task
1. **Model Architecture:**  
   - Used pre-trained DenseNet-161 and ResNet-50.
   - Made an ensemble model using DenseNet-161 and ResNet-18.(Best performing model)
   - Modified the final classifier to include linear, ReLU, batch norm, dropout, and output layers for 3 classes.  

2. **Fine-Tuning Strategy:**  
   - **Progressive unfreezing** from the classifier to earlier layers, one section at a time.  
   - Epochs proportional to the number of parameters in each section.  
   - StepLR scheduler with a step size of 8 for learning rate decay.  

3. **Training:**  
   - Optimizer: Adam with `lr=1e-4`.  
   - Saved model checkpoints every 2 epochs.  
   - Logged metrics using Weights & Biases (wandb).
------
#### Specific Task VI.A
#### **2.1 Pretraining Phase (Unsupervised Learning)**

-   **Model:** Masked Autoencoder (MAE)
    
    -   **Encoder:**
        
        -   Uses **ViT-inspired architecture** with convolution-based patch embeddings.
            
        -   Image size: **64x64**, Patch size: **8x8**
            
    -   **Decoder:**
        
        -   Uses **up-convolutions** (transposed convolutions) for better reconstruction quality.
            
    -   **Masking Strategy:**
        
        -   **Pixel-wise masking** directly applied to the image.
            
        -   Masking regions with **more informative pixels** to challenge the reconstruction.
            
-   **Loss Function:** Mean Squared Error (MSE) between the reconstructed and original images.

#### **2.2 Fine-tuning Phase (Supervised Learning)**

-   **Model:**
        
    -   Adds a **classification head** consisting of:
        
        -   Global average pooling over patch embeddings.
            
        -   Fully connected layers: **512 → 256 → 3 (softmax)**
    -   finetunes the **pre-trained MAE encoder with classification head** with frozen weights for 10 epochs amd them unfreezes whole model to train
----
#### Specific Task VI.B

#### **2.1 Model Architecture:**

-   **Base Model:**
    
    -   Uses the **pre-trained MAE encoder** from Task VI.A as a **feature extractor**.
        
-   **Decoder:**
    
    -   Uses **up-convolutions** to upsample the low-resolution image to high resolution.
        
    -   Final convolution layer to **match the HR output size (150x150)** which also works as smoothing.

---
### model weights
[Final model weights](https://www.kaggle.com/models/akhilblanc/model-weights/)
epochs specific weights are as follows
[Common task](https://www.kaggle.com/models/akhilblanc/lens_classifcation/)
[pretrained_mae](https://www.kaggle.com/models/akhilblanc/pretrained-mae/)
---
### Results
#### Common Task
- Accuracy: 95.92%

![ROC-AUC score](https://github.com/user-attachments/assets/671ab52d-117d-4f96-8be3-b56512604686)

#### Specific Task VI.A
-**Pretraining Phase**
![image](https://github.com/user-attachments/assets/b9d80ad0-bcae-4525-9229-4ab854bea102)
-**Finetuned Classifier**
Accuracy: 81.64%

![ROC-AUC score](https://github.com/user-attachments/assets/7ad0df04-66de-4558-ae2e-698c9ac41e80)

#### Specific Task VI.B
Test PSNR: 38.631875 , SSIM: 0.95930625, MSE: 0.00013130609884337034


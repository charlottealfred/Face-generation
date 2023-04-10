# Face generation
## Introduction
* This project built a GUI for pre-trained InterfaceGAN and CycleGAN
* This GUI has two main functions, generating faces and changing image styles

* Run command：
```cmd
$ python gan_gui.py
```

* Table widget 1 : Generate new faces：
![Table widget 1 : Generate new faces](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/Table%20widget%201%20Generate%20new%20faces.png "Table widget 1 : Generate new faces")

* Function：
    * Generate two human faces by using 'random' or 'input seed number' noise seed
    * The available models are 'pggan_celebahq', 'stylegan_celebahq', 'stylegan_ffhq'
    * The available latent space for 'stylegan_celebahq', 'stylegan_ffhq' are 'W', 'Z'.
    * Aftering choosing the parameters, click the 'Change original faces' button to generate faces.
    * Moing the silider to change the attributes of generated faces.
    
![Change faces](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/change_faces.png "Change faces")

   * Clicking the save photo button to save faces.
   * The format of the default picture name is:         
   'picture0_age_gender_eyeglasses_smile_pose_noise seed_model name_latent space type'. 
   For example the name 'picture1_a0_g0_e0_s0_p0_sd0_mostylegan_ffhq_lsW' means this is the second picture which is generated in tab 1 with parameter age = 0, gender = 0 eyeglasses = 0, smile = 0, pose = 0, noise seed = 0, model name = stylegan_ffhq,latent space type = W, pagan model does not have latent space

![Save](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/save_tab1.png "Save")   

   
* Table widget 2: Change style：

![Table widget 2: Change style](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/Table%20widget%202%20Change%20style.png "Table widget 2: Change style")

* Function：
    * Changing the style of uploading image.
    * Choosing the style you want.
    * Clicking the zoom in button to view the style changed picture full screen
 ![Zoom in](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/zoom_in.png "Zoom in")
 
    * Clicking the save photo button to save faces.
    * The format of the default picture name is original file name add style mane, such as XXXX_monet.png
![Save](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/save_tab2.png "Save in")    
    
    
## How to download this project

### Download the entire project directly from Google Drive
   * Google Drive: https://drive.google.com/drive/folders/1TeST5IExNgV9MV8741lGN4nN_R7PQcR6?usp=share_link
### Clone project 
   * First clone this project from github
   * Download pretrain-models
      * InterfaceGAN: 
         * download pretrained models from the InterfaceGAN project
         * Url: https://colab.research.google.com/github/genforce/interfacegan/blob/master/docs/InterFaceGAN.ipynb
                  ![InterfaceGAN colab](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/interfacegan_colab.png "InterfaceGAN colab")

         * These pretrained models can also be downloaded from my Google Drive
         * Url: https://drive.google.com/drive/folders/19-_NH7_GVRa_ywtlMgShu7Me4QfkvM4w?usp=share_link
         
         * Put these models in this floder:
      * CycleGAN：
         * download pretrained models from the Cycle project
         * The models are: style_monet, style_cezanne, style_ukiyoe, style_vangogh
         * Url: https://colab.research.google.com/github/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/CycleGAN.ipynb#scrollTo=gdUz4116xhpm
                  ![CycleGAN colab](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/cyclegan_colab.png "CycleGAN colab")

         * These pretrained models can also be downloaded from my Google Drive
         * Url: https://drive.google.com/drive/folders/1VLLOmDOpeq7luHusK3QzEuAv8-z1F24J?usp=share_link
         
         * Store models in checkpoint folder and put these models in Put the model into the corresponding folder
         ![checkpoint folder](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/checkpoints.png "checkpoint folder")
         
         
 ## Reference
   * InterfaceGAN

   * CycleGAN
   * Please note that I have made some changes to the test.py
 ![test.py change](https://github.com/charlottealfred/Face-generation/blob/main/readme_picture/testpy_change.png "test.py change")
    


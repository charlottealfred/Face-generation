# Face generation
## Introduction
* This project built a GUI for pre-trained InterfaceGAN and CycleGAN
* This interface has two main functions, generating faces and changing image styles

* Run command：
```cmd
$ python gan_gui.py
```

* Table widget 1 :Generate new faces：

![省级示例](https://raw.githubusercontent.com/snakejordan/static-file/master/administrative-divisions-of-China-on-Python/doc/images/api_example_province.png "省级示例")

* Function：
    * Generate two human faces by using random or input seed number noise seed
    *  The available models are 'pggan_celebahq', 'stylegan_celebahq', 'stylegan_ffhq'
    * The available latent space for 'stylegan_celebahq', 'stylegan_ffhq' are 'W', 'Z'.
    *Aftering choosing the parameters, click the 'Change original faces' button to generate faces.
![省级示例](https://raw.githubusercontent.com/snakejordan/static-file/master/administrative-divisions-of-China-on-Python/doc/images/api_example_province.png "省级示例")
    * Moing the silider to change the attributes of generated faces.
![省级示例](https://raw.githubusercontent.com/snakejordan/static-file/master/administrative-divisions-of-China-on-Python/doc/images/api_example_province.png "省级示例")   
   * Clicking the save photo button to save faces.
   * The format of the default picture name is:         'picture0_age_gender_eyeglasses_smile_pose_noise seed_model name_latent space type'. 
   For example the name 'picture1_a0_g0_e0_s0_p0_sd0_mostylegan_ffhq_lsW' means this is the second picture which is generated in tab 1 with parameter age = 0, gender = 0 eyeglasses = 0, smile = 0, pose = 0, noise seed = 0, model name = stylegan_ffhq,latent space type = W, pagan model does not have latent space

![省级示例](https://raw.githubusercontent.com/snakejordan/static-file/master/administrative-divisions-of-China-on-Python/doc/images/api_example_province.png "省级示例")   

   
* Table widget 2: Change style：

![省级示例](https://raw.githubusercontent.com/snakejordan/static-file/master/administrative-divisions-of-China-on-Python/doc/images/api_example_province.png "省级示例")

* Function：
    * Changing the style of uploading image.
    * Choosing the style you want.
    * Clicking the zoom in button to view the style changed picture full screen
 ![省级示例](https://raw.githubusercontent.com/snakejordan/static-file/master/administrative-divisions-of-China-on-Python/doc/images/api_example_province.png "省级示例")
 
    * Clicking the save photo button to save faces.
    * The format of the default picture name is original file name add style mane, such as XXXX_monet.png
    
    
    
## How to download this project

### Download the entire project directly from Google Drive
   * Google Drive: 
### Clone project 
   * First clone this project from github
   * Download pretrain-models
      * InterfaceGAN: 
         * download pretrained models from the InterfaceGAN project
         * Url:
         * These pretrained models can also be downloaded from my Google Drive
         * Url:
         
         * Put these models in this floder:
      * CycleGAN：
         * download pretrained models from the Cycle project
         * The models are: 
         * Url:
         * These pretrained models can also be downloaded from my Google Drive
         * Url:
         
         * Store models in checkpoint
         
         * Put these models in Put the model into the corresponding folder
         
         
 ## Reference
   * InterfaceGAN

   * CycleGAN
   * Please note that I have made some changes to the test.py
 ![省级示例](https://raw.githubusercontent.com/snakejordan/static-file/master/administrative-divisions-of-China-on-Python/doc/images/api_example_province.png "省级示例")
    


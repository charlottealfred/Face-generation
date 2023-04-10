import random
import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic, QtGui, QtWidgets
import os.path
import numpy as np
import cv2
import PIL.Image
import torch
from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
import os
import traceback

# check the tensorflow version and GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
version=tf.__version__  #check tensorflow version
gpu_ok=tf.test.is_gpu_available()  #check gpu
print("tf version:",version,"\nuse GPU:",gpu_ok)
tf.test.is_built_with_cuda()  # check GUDA


# This class is used for pop-ups when saving files
class Save_file_dia(QDialog):
    """Employee dialog."""
    def __init__(self, parent=None):
        super().__init__(parent)
        # Load the dialog's GUI
        uic.loadUi("save_file_dia.ui", self)

        self.path = '' #path for saving files

        '''The following eight properties are used to store data from the main window, 
        which will be used for file saving'''
        self.age = 0
        self.gender= 0
        self.eyeglasses= 0
        self.pose= 0
        self.smile= 0
        self.noise_seed = 0
        # model type: ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
        self.model_name = 0
        self.latent_space_type = 0

        # pushButton_cp is used for choosing the path of saved photo
        self.pushButton_cp.clicked.connect(self.choose_path)

        # pushButton_save is used for saving photos
        self.pushButton_save.clicked.connect(self.save_pic)
        # self.pushButton_save.setEnabled(False)

        # comboBox_2 is used for choosing format of saving photos
        picture_format = ['png', 'jpg', 'jpeg']
        self.comboBox_2.addItems(picture_format)

        # comboBox is used to select the images that can be stored,
        # 'choose a picture' is the default text for comboBox
        picture_numbers = ['choose a picture']
        # self.comboBox_n.addItems(list_number_of_photo)
        self.comboBox.addItems(picture_numbers)

        # When users choose one item in the comboBox, the method click_combobox will be activated,
        # it will set the file name
        self.comboBox.activated.connect(self.click_combobox)

        self.label_status.setText('Please select the file you want to store and the file storage path.')


        # lineEdit_2 is used for changing the file name, the initial value is ‘None’
        self.lineEdit_2.setText('None')

        # file_name is used to store the original name of the file
        self.file_name = ''

        # tab2_temp is used to store the name of the selected style, the optional styles are
        # Monet, Van Gogh, Cézanne, Ukiyo - e
        self.tab2_temp = ''



    '''This method is used for choosing path
        if users do not select a path, the text in lineEdit will be 'No path' and
        the file can not be save'''
    def choose_path(self):


        self.path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
        if self.path:
            self.lineEdit.setText(f'{self.path}')
            print(self.path)

        else:
            self.lineEdit.setText('No path!')



    '''The click_combobox method is used for setting the name of picture which will be saved
            if users choose '1' or '2' in comboBox , the default picture name will be picture0 or picture1 and some parameters from tab 1
            users can change the name in lineEdit_2
            The format of the image name is:
                picture0_age_gender_eyeglasses_smile_pose_noise seed_model name_latent space type
                For example the name 'picture1_a0_g0_e0_s0_p0_sd0_mostylegan_ffhq_lsW' means
                this is the second picture which is generated in tab 1 with parameter age = 0, gender = 0
                eyeglasses = 0, smile = 0, pose = 0, noise seed = 0, model name = stylegan_ffhq,
                latent space type = W
                pagan model does not have latent space

                if users choose 'style_changed_photo' in comboBox, the default name will be the original name
                add style mane, such as XXXX_monet.png
                '''
    def click_combobox(self, info):
        # w = self.win
        print(self.comboBox.currentText())
        # if self.path == 'No path!'
        print('para of main windows', self.age, self.gender, self.noise_seed)

        if self.comboBox.currentText() == "1":
            print('-----1------')
            if self.model_name == 'pggan_celebahq': #pagan model does not have latent space
                self.lineEdit_2.setText(f'picture0_a{self.age}_g{self.gender}_e{self.eyeglasses}_s{self.smile}'
                                        f'_p{self.pose}_sd{self.noise_seed}'
                                        f'_mo{self.model_name}')

            else:
                self.lineEdit_2.setText(f'picture0_a{self.age}_g{self.gender}_e{self.eyeglasses}_s{self.smile}'
                                        f'_p{self.pose}_sd{self.noise_seed}'
                                        f'_mo{self.model_name}_ls{self.latent_space_type}')

        if self.comboBox.currentText() == "2":
            print('-----2------')
            if self.model_name == 'pggan_celebahq':
                self.lineEdit_2.setText(f'picture0_a{self.age}_g{self.gender}_e{self.eyeglasses}_s{self.smile}'
                                        f'_p{self.pose}_sd{self.noise_seed}'
                                        f'_mo{self.model_name}')
            else:
                self.lineEdit_2.setText(f'picture1_a{self.age}_g{self.gender}_e{self.eyeglasses}_s{self.smile}'
                                        f'_p{self.pose}_sd{self.noise_seed}'
                                        f'_mo{self.model_name}_ls{self.latent_space_type}')

        if self.comboBox.currentText() == 'style_changed_photo':
            # if self.tab == 2:
            print('-----3------')
            print(f'{self.file_name}_{self.tab2_temp}')

            self.lineEdit_2.setText(f'{self.file_name}_{self.tab2_temp}')

        self.label_status.setText(f'Default file name: {self.lineEdit_2.text()}')
    '''This method is used for image storage 
        if self.comboBox.currentText() == 'choose a picture' prompts the user to choose 
        a picture in comboBox

        if self.path is empty prompts the user to choose a path

        tab 1 'Generate new faces' can generate two pictures, they are named picture0.png and picture1.png
        if users choose '1' in comboBox, picture0.png will be saved, 
        if users choose '2' in comboBox, picture1.png will be saved

        tab 2 'Change style' can change the style of the picture which users uploaded, the style changed picture
        will be saved as 'cyclegan_result.png'
        if users choose 'style_changed_photo' in comboBox, cyclegan_result.png will be saved

        the text in conmoBox will change with the current tab
        for tab 1, the text will be ['choose a picture', '1', '2']
        for tab 2, the text will be ['choose a picture', 'style_changed_photo']
    '''

    def save_pic(self):
        # w = self.win
        if self.comboBox.currentText() == 'choose a picture':
            self.label_status.setText('Please choose a picture')

        if self.path == '':
            self.label_status.setText('Please choose a path')

        if (self.comboBox.currentText() != 'choose a picture') and \
                (self.path != ''):
            if self.comboBox.currentText() == "1":
                print('-----------')
                pic0 = PIL.Image.open("picture0.png")
                # save a image using extension
                pic0.save(f"{self.path}/{self.lineEdit_2.text()}.{self.comboBox_2.currentText()}")
                # self.statusBar.showMessage(f'Picture {self.save_pic_num} saved', 0)
            if self.comboBox.currentText() == "2":
                pic1 = PIL.Image.open("picture1.png")
                # save a image using extension
                pic1.save(f"{self.path}/{self.lineEdit_2.text()}.{self.comboBox_2.currentText()}")
                # self.statusBar.showMessage(f'Picture {self.save_pic_num} saved', 0)

            if self.comboBox.currentText() == 'style_changed_photo':
                print('-----3------')
                pic2 = PIL.Image.open("cyclegan_result.png")
                # save a image using extension
                pic2.save(f"{self.path}/{self.lineEdit_2.text()}.{self.comboBox_2.currentText()}")
                self.label_status.setText(f'{self.lineEdit_2.text()}.{self.comboBox_2.currentText()} saved!')
            self.label_status.setText(f'{self.lineEdit_2.text()}.{self.comboBox_2.currentText()} saved!')

class MyGUI(QMainWindow):


    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi('gan gui.ui', self) # load ui file
        # self.show()

        # The following are some of the initial model parameters
        # '------------------------------------------------------------------------'
        #Image size is 512*512
        self.viz_size = 512

        # There are three models can be choosed to generate faces, default model is stylegan_ffhq
        # model type: ['pggan_celebahq','stylegan_celebahq', 'stylegan_ffhq']
        self.model_name = "stylegan_ffhq"


        self.latent_space_type = "W"# latent_space_type of model 'stylegan_celebahq', 'stylegan_ffhq'

        self.num_samples = 2  #Number of images that can be generated by the model
        self.noise_seed = 0
        # self.images = 'default.jpg'
        self.boundaries = {}
        self.synthesis_kwargs = {}

        # attributes of faces
        self.age = 0
        self.gender = 0
        self.eyeglasses = 0
        self.smile = 0
        self.pose = 0

        self.tab2_temp = ''
        self.file_name = ''

        # '------------------------------------------------------------------------'

        # Setting of Tab 1
        # '------------------------------------------------------------------------'


        # set the default image in tab 1, current_file shows the generated faces
        # The initial image is the instructions for using tab1.
        self.current_file = 'tab1_instr.png'
        # let picture resize with windows
        pixmap = QtGui.QPixmap(self.current_file)
        pixmap = pixmap.scaled(self.width(), self.height())
        self.label.setPixmap(pixmap)
        self.label.setMinimumSize(1,1)

        #Step 1: choose model and parameters

        # users can choose to use random noise seed or fixed noise seed,
        # When one of the radioButton is selected, the other one cannot be selected
        self.radioButton_rand.toggled.connect(self.random_seed)
        self.radioButton_seedn.toggled.connect(self.choose_seed)
        # self.lineEdit_seedn.setEnabled(False)
        # self.lineEdit_seedn.setEnabled(False)
        self.lineEdit_seedn.setText('')
        self.lineEdit_seedn.returnPressed.connect(self.enter_seed_number)

        #set values of comboBoxs
        list_latent_space = ['W', 'Z']
        list_model_name = ['stylegan_ffhq', 'stylegan_celebahq', 'pggan_celebahq']
        self.comboBox_l.addItems(list_latent_space) # comboBox for choosing latent space
        self.comboBox_m.addItems(list_model_name) # comboBox for choosing model

        self.comboBox_l.activated.connect(self.click_combobox_l)
        self.comboBox_m.activated.connect(self.click_combobox_m)


        # Step 2: click bottom change face to generate two faces,
        # save face buttom and sliders will be activated after this button clicked
        self.pushButton_changeof.clicked.connect(self.sample_image)

        # save faces
        self.pushButton_savep.clicked.connect(self.save_file_tab1)
        self.pushButton_savep.setEnabled(False)

        # Step 3: using sliders to change the age, gender, eyeglasses, smile, pose of generated faces
        # These five sliders can change the attributes of generated faces,
        self.slider_age.setMinimum(-3.0)
        self.slider_age.setMaximum(3.0)
        self.slider_age.setEnabled(False)

        self.slider_gender.setMinimum(-3.0)
        self.slider_gender.setMaximum(3.0)
        self.slider_gender.setEnabled(False)

        self.slider_eyeglasses.setMinimum(-2.9)
        self.slider_eyeglasses.setMaximum(3.0)
        self.slider_eyeglasses.setEnabled(False)

        self.slider_smile.setMinimum(-3.0)
        self.slider_smile.setMaximum(3.0)
        self.slider_smile.setEnabled(False)

        self.slider_pose.setMinimum(-3.0)
        self.slider_pose.setMaximum(3.0)
        self.slider_pose.setEnabled(False)

        # Move The Slider

        self.slider_age.valueChanged.connect(self.slide_it_age)
        self.slider_gender.valueChanged.connect(self.slide_it_gender)
        self.slider_eyeglasses.valueChanged.connect(self.slide_it_eyeglasses)
        self.slider_smile.valueChanged.connect(self.slide_it_smile)
        self.slider_pose.valueChanged.connect(self.slide_it_pose)

        # '------------------------------------------------------------------------'

        # Setting of Tab 2
        # '------------------------------------------------------------------------'

        # set the default image in tab 2,
        # current_file1 is the original photo and current_file2 is the style changed photo
        self.current_file1 = 'tab2_1.png'
        pixmap1 = QtGui.QPixmap(self.current_file1)
        pixmap1 = pixmap1.scaled(self.width(), self.height())
        self.label_2.setPixmap(pixmap1)
        self.label_2.setMinimumSize(1, 1)

        self.current_file2 = 'tab2_2.png'
        pixmap2 = QtGui.QPixmap(self.current_file2)
        pixmap2 = pixmap2.scaled(self.width(), self.height())
        self.label_3.setPixmap(pixmap2)
        self.label_3.setMinimumSize(1, 1)



        # Step 1: users need to upload a photo
        # this button will activate the change style button
        self.pushButton_up.clicked.connect(self.upload_picture)

        # Step 2: users need to choose the style of photo, the default style is monet
        self.radioButton_mo.toggled.connect(self.choose_style)
        self.radioButton_mo.setChecked(True)
        self.radioButton_van.toggled.connect(self.choose_style)
        self.radioButton_cez.toggled.connect(self.choose_style)
        self.radioButton_uki.toggled.connect(self.choose_style)

        # Step 3: click the change style button to change the style of photo
        # this button will activated the save photo button and zoom in photo button
        self.pushButton_cs.clicked.connect(self.change_style)
        self.pushButton_cs.setEnabled(False)

        # Step 4: click the save photo button to save photo or
        # zoom in photo button to view style changed photo full screen
        self.pushButton_zp.clicked.connect(self.zoom_in_photo)
        self.pushButton_zp.setEnabled(False)

        self.pushButton_sp.clicked.connect(self.save_file_tab2)
        self.pushButton_sp.setEnabled(False)


        # self.progressBar.clicked.connect(self.)

    # '------------------------------------------------------------------------'

    # Methods of Tab 1
    # '------------------------------------------------------------------------'

    '''The choose_seed and random_seed methods are mutually exclusive, 
        and users can only select the noise seed by one method.
        When one method is enabled, the lineEdit of the other method is deactivated'''

    def random_seed(self):
        # self.enter_seed = 0
        self.statusBar.showMessage('Using random seed', 0)
        self.lineEdit_seedn.clear()
        self.lineEdit_seedn.setEnabled(False)
        self.lineEdit_random.setEnabled(True)
        self.noise_seed = random.randint(0,1000)
        self.lineEdit_random.setText(f'{self.noise_seed}')
        print('lindEdit_2.setText', self.noise_seed)

    def choose_seed(self):
        self.statusBar.showMessage('Please enter an integer less than 1000 and press Enter', 0)
        self.lineEdit_seedn.setEnabled(True)
        self.lineEdit_random.setEnabled(False)
        self.lineEdit_random.setText('')
        # self.enter_seed = 1


    '''This method checks if the input matches the requirements
        The input need to be an integer less than 1000'''
    def enter_seed_number(self):
        if self.lineEdit_seedn.text() == '':
            print('noise seed empty')
            self.statusBar.showMessage('Please enter noise seed number', 0)
            self.lineEdit_seedn.setText('')
            self.noise_seed = 0

        if self.lineEdit_seedn.text().isdigit() == False:
            print('noise seed false')
            self.statusBar.showMessage('Please enter an integer', 0)
            self.lineEdit_seedn.setText('')
            self.noise_seed = 0

        if self.lineEdit_seedn.text().isdigit() == True:
            # evaluate whether the input is less than 1000
            if int(self.lineEdit_seedn.text()) <= 1000:
                print(f'Successfully entered noise seed: {self.lineEdit_seedn.text()}')
                self.statusBar.showMessage(f'Successfully entered noise seed: {self.lineEdit_seedn.text()}', 0)
                self.noise_seed = int(self.lineEdit_seedn.text())

            else:
                self.statusBar.showMessage('The input integer must be less than 1000', 0)
                self.lineEdit_seedn.setText('')
                self.noise_seed = 0

    # The method of click_combobox_m which allow users to choose model
    # pggan_celebahq does not have latent space option
    def click_combobox_m(self):
        print(self.comboBox_m.currentText(), type(self.comboBox_m.currentText()))
        self.model_name = self.comboBox_m.currentText()
        self.statusBar.showMessage(f'Using model: {self.comboBox_m.currentText()}', 0)
        if self.model_name == 'pggan_celebahq':
            self.comboBox_l.setEnabled(False)
        else:
            self.comboBox_l.setEnabled(True)


    # The method of comboBox_l which allow users to choose latent space type
    def click_combobox_l(self):
        print(self.comboBox_l.currentText())
        self.latent_space_type = self.comboBox_l.currentText()
        self.statusBar.showMessage(f'Using latent space: {self.comboBox_l.currentText()}', 0)



    '''The following five methods allow users to adjust the generated faces through the slider'''
    def slide_it_age(self, value):
        # if num_label == 1:
        self.age = value
        self.label_age.setText('age = ' + str(value))
        self.change_face()

    def slide_it_gender(self, value):
        # if num_label == 1:
        self.gender = value
        self.label_gender.setText('gender = ' + str(value))
        self.change_face()


    def slide_it_eyeglasses(self, value):
        # if num_label == 1:
        self.eyeglasses = value
        self.label_eyeglasses.setText('eyeglasses = ' + str(value))
        self.change_face()

    def slide_it_smile(self, value):
        # if num_label == 1:
        self.smile = value
        self.label_smile.setText('smile = ' + str(value))
        self.change_face()

    def slide_it_pose(self, value):
        # if num_label == 1:
        self.pose = value
        self.label_pose.setText('pose = ' + str(value))
        self.change_face()

    # reset 5 sliders
    def slide_reset(self):
        self.slider_age.setValue(0)
        self.slider_gender.setValue(0)
        self.slider_eyeglasses.setValue(0)
        self.slider_smile.setValue(0)
        self.slider_pose.setValue(0)

    # '------------------------------------------------------------------------'
    # Some methods of interfaceGan

    def build_generator(self):
        """Builds the generator by model name."""
        self.statusBar.showMessage(f'Building Generator {self.model_name}', 0)
        print('build ',self.model_name)
        gan_type = MODEL_POOL[self.model_name]['gan_type']
        if gan_type == 'pggan':
            generator = PGGANGenerator(self.model_name)
        elif gan_type == 'stylegan':
            generator = StyleGANGenerator(self.model_name)
        return generator

    def sample_codes(self):

        """Samples latent codes randomly."""
        np.random.seed(self.noise_seed)
        codes = self.generator.easy_sample(self.num_samples)
        if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
            codes = torch.from_numpy(codes).type(torch.FloatTensor).to(self.generator.run_device)
            codes = self.generator.get_value(self.generator.model.mapping(codes))
        return codes


    '''This method has two functions, first combine the two generated face images 
        and then store the two images as 'picture0.png' and 'picture1.png' respectively
        
        For example, if users choose to generate two face images, these two images will be 
        stitched together into one row and two columns, 
        combined_image[0:0 + 512, 0:0 + 512] for picture0,
        combined_image[512:512 + 512, 0:0 + 512] for picture1'''
    def imshow(self,image):
        """Shows images in one figure."""

        viz_size = self.viz_size
        num, height, width, channels = image.shape

        col = 2
        assert num % col == 0
        row = num // col

        combined_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

        for idx, image in enumerate(image):
            i, j = divmod(idx, col)
            y = i * viz_size
            x = j * viz_size
            if height != viz_size or width != viz_size:
                image = cv2.resize(image, (viz_size, viz_size))
            combined_image[y:y + viz_size, x:x + viz_size] = image # combine the two generated face images
            PIL.Image.fromarray(image).save(f'picture{idx}.png') # store the two images respectively

        return combined_image # return the combined image


    '''This method can generate two face pictures according to the parameters set by the user, 
        the generated face pictures are combined and displayed in the label of tab1, 
        this combined picture is saved as "sample_face_n.png", 
        after the above steps are completed five sliders and save save photo button will be enabled'''
    def sample_image(self):
        self.statusBar.showMessage(f'Generating faces using model: {self.model_name} , '
                                   f'noise seed: {self.noise_seed}', 0)
        print(f'Noise seed: {self.noise_seed}')
        # ---------------------generate faces------------------------
        ATTRS = ['age', 'eyeglasses', 'gender', 'pose', 'smile']
        self.generator = self.build_generator()

        # load boundary files, they will be use to
        # adjust the latent code in the change face() method
        for i, attr_name in enumerate(ATTRS):
            boundary_name = f'{self.model_name}_{attr_name}'
            if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
                self.boundaries[attr_name] = np.load(
                    f'boundaries/{boundary_name}_w_boundary.npy')
            else:
                self.boundaries[attr_name] = np.load(
                    f'boundaries/{boundary_name}_boundary.npy')


        self.latent_codes = self.sample_codes() # sample latent code
        if self.generator.gan_type == 'stylegan' and self.latent_space_type == 'W':
            self.synthesis_kwargs = {'latent_space_type': 'W'}
        else:
            self.synthesis_kwargs = {}

        self.images = self.generator.easy_synthesize(self.latent_codes, **self.synthesis_kwargs)['image']
        # ---------------------generate faces------------------------



        # plt.imshow(imshow(images, col=num_samples))
        # plt.show()

        
        im = PIL.Image.fromarray(self.imshow(self.images)) # combine generated faces
        im.save("sample_face_n.png")
        self.current_file = "sample_face_n.png"
        print('------------------1---------------------')

        pixmap = QtGui.QPixmap(self.current_file) # show combined faces in label of tab1
        pixmap = pixmap.scaled(self.width(), self.height())
        self.label.setPixmap(pixmap)

        print('------------------2---------------------')


        # enable sliders and save photo button
        self.slider_age.setEnabled(True)
        self.slider_gender.setEnabled(True)
        self.slider_eyeglasses.setEnabled(True)
        self.slider_pose.setEnabled(True)
        self.slider_smile.setEnabled(True)
        self.pushButton_savep.setEnabled(True)

        self.slide_reset()

        torch.cuda.empty_cache()
        self.statusBar.showMessage(f'Faces generated using model: {self.model_name} , '
                                   f'noise seed: {self.noise_seed}', 0)


    '''When the user adjusts the face image parameters using sliders, 
        this method will adjust the face according to the parameters
        this combined changed picture is saved as "change_face.png" and shown in label of tab1, 
        '''
    def change_face(self):
        self.statusBar.showMessage(f'Changing faces', 0)
        # face attributes
        ATTRS1 = {'age': self.age, 'eyeglasses': self.eyeglasses,
                  'gender': self.gender, 'pose': self.pose, 'smile': self.smile}

        new_codes = self.latent_codes.copy()
        for i, attr_name in enumerate(ATTRS1):  # change the face by changing the latent code
            new_codes += self.boundaries[attr_name] * ATTRS1[attr_name]

        self.new_images = self.generator.easy_synthesize(new_codes, **self.synthesis_kwargs)['image']
        print('------------------3---------------------')
        # plt.imshow(imshow(new_images, col=num_samples))
        # plt.show()

        # self.current_file = self.imshow(new_images, col=num_samples)

        imn = PIL.Image.fromarray(self.imshow(self.new_images)) # combine changed faces
        imn.save("change_face.png")
        self.current_file = "change_face.png"


        pixmap = QtGui.QPixmap(self.current_file)
        pixmap = pixmap.scaled(self.width(), self.height()) # show combined faces in label of tab1
        self.label.setPixmap(pixmap)

        torch.cuda.empty_cache()

        print('------------------4---------------------')
        self.statusBar.showMessage(f'Faces changed', 0)

    ''' This method will open the save file dialog window
            pass attributes of faces to dialog window and 
            set the content of the picture number comboBox '''
    def save_file_tab1(self):

        sf = Save_file_dia()
        sf.age = self.age
        sf.gender = self.gender
        sf.eyeglasses = self.eyeglasses
        sf.pose = self.pose
        sf.smile = self.smile
        sf.noise_seed = self.noise_seed
        sf.model_name = self.model_name
        sf.latent_space_type = self.latent_space_type

        picture_numbers = ['choose a picture', '1', '2']
        # self.comboBox_n.addItems(list_number_of_photo)
        sf.comboBox.clear()
        sf.comboBox.addItems(picture_numbers)
        sf.exec()

    # '------------------------------------------------------------------------'

    # Methods of Tab 2
    # '------------------------------------------------------------------------'

    '''The method of pushButton_up
       Let users upload a picture and show the picture in tab2 label_2
        and then save the image file as image.png to the 'datasets\\test_pic' folder
       self.file_name save the file name'''
    def upload_picture(self):
        fname = QFileDialog.getOpenFileName(self, 'All Files(*)')  # open File Explorer
        if fname[0]:
            self.pushButton_cs.setEnabled(True)
            # file name with extension
            file_name = os.path.basename(fname[0])

            # file name without extension
            print(os.path.splitext(file_name)[0])
            self.file_name = os.path.splitext(file_name)[0]
            self.statusBar.showMessage(f'{self.file_name} uploaded', 0)
            # print(fname[0],'\n',os.path.splitext(fname[0])[0])

            # show the image in label_2
            pixmap = QtGui.QPixmap(fname[0])
            pixmap = pixmap.scaled(self.width(), self.height())
            self.label_2.setPixmap(pixmap)

            im1 = PIL.Image.open(fname[0])
            # save a image using extension
            image_path = 'datasets\\test_pic'
            im1.save(f"{image_path}/image.png")
        else:
            self.statusBar.showMessage('Please upload a picture', 0)


    # The four radioButton's correspond to the four styles
    # This method check which radioButton is clicked and save the style in self.tab2_temp
    def choose_style(self):
        sender = self.sender()  # return the ratioButton clicked

        if sender == self.radioButton_mo:
            if sender.isChecked():
                print('Monet')
                self.tab2_temp = 'monet'


        if sender == self.radioButton_van:
            if sender.isChecked():
                print('Vangogh')
                self.tab2_temp = 'vangogh'


        if sender == self.radioButton_cez:
            if sender.isChecked():
                print('Cezanne')
                self.tab2_temp = 'cezanne'

        if sender == self.radioButton_uki:
            if sender.isChecked():
                print('Ukiyo-e')
                self.tab2_temp = 'ukiyoe'

    '''This function runs the command com, which converts the images inside 
        the folder 'datasets\\test_pic' to the style in self.tab2_temp
        and saves the new image file as 'cyclegan_result.png' '''
    def change_style(self):
        self.statusBar.showMessage(f'Changing style', 0)
        com = 'python test.py --dataroot datasets\\test_pic '+\
              '--name style_'+self.tab2_temp+'_pretrained '+\
              '--model test --preprocess none  --no_dropout'

        os.system(com)

        # show the image in label_3
        pixmap = QtGui.QPixmap(f"cyclegan_result.png")
        pixmap = pixmap.scaled(self.width(), self.height())
        self.label_3.setPixmap(pixmap)

        # after changed the style of the image, activate the zoom in photo button
        # and save photo button to check and save style changed photo
        self.pushButton_sp.setEnabled(True)
        self.pushButton_zp.setEnabled(True)
        # self.probar(1)
        self.statusBar.showMessage(f'Style changed', 0)


    # The method of pushButton_zp
    # view style changed photo full screen
    def zoom_in_photo(self):
        im = PIL.Image.open('cyclegan_result.png')
        im.show()


    ''' This method will open the save file dialog window
        pass file_name, tab2_temp to dialog window and 
        set the content of the picture number comboBox '''
    def save_file_tab2(self):


        sf = Save_file_dia()
        sf.file_name = self.file_name
        sf.tab2_temp = self.tab2_temp
        sf.tab = 2

        # change the content in the comboBox of save file dialog
        picture_numbers = ['choose a picture', 'style_changed_photo']
        sf.comboBox.clear()
        sf.comboBox.addItems(picture_numbers)

        sf.exec()


    # '------------------------------------------------------------------------'




    def raise_error(self):
        assert False



def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("error catched!:")
    print("error message:\n", tb)
    QtWidgets.QApplication.quit()




def main():
    sys.excepthook = excepthook
    app = QApplication([])
    window = MyGUI()
    window.show()
    ret = app.exec_()
    print("event loop exited")
    sys.exit(ret)


if __name__ == '__main__':
    main()
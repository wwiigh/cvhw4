# NYCU Computer Vision 2025 Sprint HW4
Student ID: 313553002  
Name: 蔡琮偉
## Introduction
 The objective of the homework is Image Restoration, to remove rain and snow effect, I use
 PromptIR as backbone. To finish the task, I use the implementation form their github repo,
 and add SSIM loss to get better performance. With the above method, the model get PSNR
 30.99.
## How to install
### Install the environment
`
conda env create -f environment.yml
`  
If there are torch install error, please go to https://pytorch.org/get-started/locally/ to get correct torch version  
If get no tensorboard error in training, please run following command  
`
conda install -c conda-forge tensorboard
`  
or you can remove all SummaryWriter function  
### Pretained model and dataset download
Pretained model: [https://drive.google.com/file/d/1MXcpnYmCLz2UPknXsEMguEUMYlxZvmwc/view?usp=sharing](https://drive.google.com/file/d/1MXcpnYmCLz2UPknXsEMguEUMYlxZvmwc/view?usp=sharing)
dataset: [https://drive.google.com/file/d/1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w/view?usp=drive_link](https://drive.google.com/file/d/1bEIU9TZVQa-AF_z6JkOKaGp4wYGnqQ8w/view?usp=drive_link)
### File structure
create model and data folder, put the model and unzip dataset to corresponding folder. It should look like this  
project-root/  
├── src/  
│   ├── pytorch_ssim/
│   │   ├── __init__.py 
│   ├── dataset.py  
|   ├── eval.py   
|   ├── model.py 
|   ├── schedulers.py
|   ├── test.py  
|   ├── train.py  
│   ├── utils.py  
├── data/       
│   ├── test/
│   ├── train/  
├── model/  
│   ├── pretained.pth     
├── README.md          
├── environment.yml    
└── .gitignore          
### How to use
For test, change the function test's parameter to the pretained model path then run  
`
python src/test.py
`  
For training, run  
`
python src/train.py
`  
## Performance snapshot
![image](https://github.com/user-attachments/assets/ef2de37c-4223-4d1f-bcb1-8d93ac5508e8)





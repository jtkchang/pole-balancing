Environmets:    
> Ubuntu 16.04 or higher
   > cuda 10+
   > use environment.yml file to create an environment for this project
        > In the terminal run command: env create -f environment.yml
        > Activate the installed environment and run the project


This is an example of how to run:

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python new_train.py --version_data 1 --version_model 1 --version_catalog 1 --num_gpu 4 --sanity_check 0 --train_batch_size 128 --test_batch_size 128 --mode train


firstargs:
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 -> explicitly mention what you want to use 

secondargs:
python new_train.py   the file with the machine learning model

thirdargs:
--version_data 1  that is indicating the version of the processed data

fourthargs:
--version_model 1  that is indicating the version of the trained model, where the trained model will be saved, also when transfer learning is implemented then this is indicating which model version is used

fifthargs:
--version_catalog 1  in which name the classification report is being saved

sixth args:
--num_gpu 4 indicating how many gpu is being used 


seventh args:
--sanity_check 0  , it will be always 0 when real experiment is being run

ninth args:
--train_batch_size 128 , this is the default batch size for training

tenth args:
--test_batch_size 128  , this is the default batch size for testing

eleventh args:
--mode train . indicating the mode, train or test




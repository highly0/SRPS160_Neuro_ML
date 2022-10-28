# SRPS160_Neuro_ML

This is a repository for final project of Neuroimaging and Machine Learning for Biomedicine course. The goal for our project is to classify between schizophrenia and control group on sMRI/fMRI data. For the dataset, we used the SRPS160 dataset (which inlcuded both fMRI and sMRI data). 

## Requirements
- numpy==1.21.2
- pandas==1.5.1
- matplotlib==3.6.1
- torch==1.12.1
- torchvision==0.13.1
- nilearn==0.9.2
- monai==1.0.0


## Getting Data
For both kinds of MRI data, we utilized a the Monai framework and Deep Learning 3d Neural network. In order to get the data to train, we utilized `ya.py`. Run the following commands to download all the neccesary files:

```bash
python3 ya.py https://disk.yandex.ru/d/rtsWt2Di1BRewA
python3 ya.py https://disk.yandex.ru/d/_Y20M_iVIwm31A
```

The first command above will download a folder with 2 files in it: `SRPBS_1600_shiza.tar.gz` and `clinica.tar.gz`. `clinica.tar.gz` will provide the labels for our MRI images. The second command will download the second half of `SRPBS_1600_shiza.tar.gz`. You shoud cat the 2 `SRPBS_1600_shiza.tar.gz` with something like this: `cat path_to_first_shiza path_to_second_shiza > data/srphs_sh_h`. 

Once you have combined the 2 shiza files (`data/srphs_sh_h`), your raw T1 sMRI data is ready to be trained. However, the same cannot be said for the fMRI data. THe fMRI data within are divided amongt a variable number of slices. We've implemented a script `get_raw_fmri.py` in order to combine the slices. Run the following command to combine the fMRI slices:

```bash
python3 get_raw_fmri.py --data_path YOUR_PATH_TO_COMBINED_SHIZA
```

That concludes the data fetching :^)

## Training

In order the run our model, use the following command:
```bash
python3 train.py
```

`train.py` comes with many different command line argument options:
- `--num_epoches`: use this flag to alter the number of epoches for training.
- `--smri`: `True` to train T1 sMRI and `False` to train fMRI
- `--preproc`: `True` to train the proprocessed version of data, `False` for raw.
- `--data_path`: the directory path for the `srphs_sh_h` file (combined shiza file)
- `--label_path`: the directory path for the `clinica` file

For example, let us say your data is at `/data/t1_linear_bfc`, your label is at `/data/clina`, to train on the raw T1 sMRI data with 10 epoches, run the following command:

```bash
python3 train.py --num_epoches 10 --smri True --preproc False --data_path /data/t1_linear_bfc --label_path /data/clina
```

Whilst the model is training, the code will automatically create plots and saved model weights (best performed val accuracy so far).

## Preprocessing

Firtly, you need to prepare your data in **BIDS** format https://bids.neuroimaging.io/. 

    Dataset/

        participant.tsv
  
        task-rest_bold.json
  
        BIDS/
  
            sub-.../
    
                anat/
      
                    sub-..._T1w.nii
        
                func/
      
                    sub-..._task-rest_bold.nii
        
To launch the fmripreproc pipeline you need to run the exact container. Load the zip **fmriprepfs_pipeline.tar.gz** with files required to build the image. To build and run a container launch inside the **fmriprepfs_pipeline** folder:

```
docker build -t fmriprep_pipeline .
docker run --platform='linux/amd64' --rm --name fmri --cpuset-cpus=0-3 -d  -v /Users/evgeniygarsiya/Desktop/NeuroML_proj/Project_dataset:/input -v /Users/evgeniygarsiya/Desktop/NeuroML_proj/Project_dataset:/output fmriprep_pipeline
```

To check progress use logs:

```
docker logs [ID]
```

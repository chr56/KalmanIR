For training and testing, we provide the directory structure. You can download the complete datasets and put thme here. 

```shell
|-- datasets
    # image SR - train
    |-- DF2K
        |-- train_HR
        |-- train_LR_bicubic
            |-- X2
            |-- X3
            |-- X4
    |-- DIV2K
        |-- train_HR
        |-- train_LR_bicubic
            |-- X2
            |-- X3
            |-- X4
    # image SR - test
    |-- benchmark
        |-- Set5
            |-- image_SRF_2
                |-- LR
                |-- HR
            |-- image_SRF_3
                |-- LR
                |-- HR
            |-- image_SRF_4
                |-- LR
                |-- HR
        |-- Set14
            |-- image_SRF_2
                |-- LR
                |-- HR
            |-- image_SRF_3
                |-- LR
                |-- HR
            |-- image_SRF_4
                |-- LR
                |-- HR
        |-- BSDS100
            |-- image_SRF_2
                |-- LR
                |-- HR
            |-- image_SRF_3
                |-- LR
                |-- HR
            |-- image_SRF_4
                |-- LR
                |-- HR
        |-- Urban100
            |-- image_SRF_2
                |-- LR
                |-- HR
            |-- image_SRF_3
                |-- LR
                |-- HR
            |-- image_SRF_4
                |-- LR
                |-- HR
        |-- Manga109
            |-- image_SRF_2
                |-- LR
                |-- HR
            |-- image_SRF_3
                |-- LR
                |-- HR
            |-- image_SRF_4
                |-- LR
                |-- HR  
    # color image denoising - train
    |-- DFWB_RGB
        |-- HQ
    # real image denoising - train & val
    |-- SIDD
        |-- train
            |-- target_crops
            |-- input_crops  
        |-- val
            |-- target_crops
            |-- input_crops 
    # grayscale JPEG compression artifact reduction - train
    |-- DFWB_CAR
        |-- HQ
        |-- LQ
            |-- 10
            |-- 30
            |-- 40  
    # gaussian color image denoising - test
    |-- ColorDN
        |-- CBSD68HQ
        |-- Kodak24HQ
        |-- McMasterHQ
        |-- Urban100HQ
    # real image denoising - test
    |-- RealDN
        |-- SIDD
            |-- ValidationGtBlocksSrgb.mat
            |-- ValidationNoisyBlocksSrgb.mat
        |-- DND
            |-- info.mat
            |-- ValidationNoisyBlocksSrgb
                |-- 0001.mat
                |-- 0002.mat
                ：  
                |-- 0050.mat
    # grayscale JPEG compression artifact reduction - test
    |-- CAR
        |-- classic5
            |-- Classic5_HQ
            |-- Classic5_LQ
                |-- 10
                |-- 30
                |-- 40
        |-- LIVE1
            |-- LIVE1_HQ
            |-- LIVE1_LQ
                |-- 10
                |-- 30
                |-- 40
```


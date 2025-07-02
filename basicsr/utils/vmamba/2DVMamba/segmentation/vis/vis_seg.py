import os

from mmseg.apis import init_model, inference_model, show_result_pyplot


# config_path = 'configs/vssm1/upernet_vssm_4xb4-160k_ade20k-512x512_tiny.py'
# checkpoint_path = ('/scratch/KurcGroup/jingwei/gpfs/checkpoint/vmamba/'
#                    'upernet_vssm_4xb4-160k_ade20k-512x512_tiny_s_iter_160000.pth')
config_path = 'configs/vssm_2d/upernet_vssm_2d_4xb4-160k_ade20k-512x512_tiny.py'
checkpoint_path = ('/gpfs/scratch/jingwezhang/result/v2dmamba/v2dmamba_fix/v2dmamba_t'
                   '/segmentation/iter_160000.pth')
img_path = 'demo/demo.png'
validataion_path = ('/scratch/KurcGroup/jingwei/Projects/VMamba/segmentation/'
                    'data/ade/ADEChallengeData2016/images/validation')
out_dir = '/gpfs/scratch/jingwezhang/result/v2dmamba/v2dmamba_fix/v2dmamba_t/segmentation/'
try:
    import segmentation.model
except:
    import model

if __name__ == '__main__':
    image_filenames = [f for f in os.listdir(validataion_path)]


# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')

# inference on given image
result = inference_model(model, img_path)

# # display the segmentation result
# vis_image = show_result_pyplot(model, img_path, result)

# save the visualization result, the output image would be found at the path `work_dirs/result.png`
vis_iamge = show_result_pyplot(model, img_path, result, out_file='work_dirs/result.png')

# # Modify the time of displaying images, note that 0 is the special value that means "forever"
# vis_image = show_result_pyplot(model, img_path, result, wait_time=5)
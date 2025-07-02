from utils.utils import WholeSlideImage
import torch
import glob
import h5py
import yaml
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', default='MaP', required=True)
parser.add_argument('--survival', action='store_true', default=False, required=True)
parser.add_argument('--slide_folder', required=True)
parser.add_argument('--h5_folder', required=True)
parser.add_argument('--heatmap_save_dir', required=True)
args = parser.parse_args()

device = torch.device('cuda')
model_path = f'{args.model_path}'
model = torch.load(model_path).to(device)
model.survival = args.survival

for path in glob.glob(args.slide_folder)[:]:
    slide_id = path.split('/')[-1][:-4]
    count_relevance = 0
    print(slide_id)
    try:
        data = h5py.File(f'{args.h5_folder}/{slide_id}.h5')
    except:
        print(f'Cannot found h5 file for: {slide_id}')
        continue
    slide_feats = torch.tensor(data['features'][:]).to(device)
    coords = torch.tensor(data['coords'][:]).to(device)

    _, _, prediction, attention, _ = model(slide_feats)
    attention = attention.cpu().detach().numpy()

    wsi = WholeSlideImage(path)
    if len(wsi.level_dim) > 3:
        vis_level = 3
    else:
        vis_level = 2
    
    heatmap = wsi.visHeatmap(
        scores=attention,
        coords=data['coords'][:],
        patch_size = (512,512),
        blur = True,
        overlap=0.0,
        cmap = 'jet',
        convert_to_percentiles = True,
        vis_level = vis_level
    )
    os.makedirs(f'{args.heatmap_save_dir}/', exist_ok=True)
    heatmap.save(f'{args.heatmap_save_dir}/{slide_id}.png')

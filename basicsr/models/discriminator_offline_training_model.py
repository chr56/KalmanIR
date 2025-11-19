from os import path as osp
from collections import OrderedDict
from typing import List, Dict, Any

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.models.util_config import read_optimizer_options
from basicsr.utils.img_util import (
    imwrite, tensor2img,
    calculate_and_padding_image,
    calculate_and_padding_image_no_overlapping,
    calculate_borders_for_chopping,
    recover_from_patches_and_remove_paddings,
)
from basicsr.utils.logger import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.validation_util import log_validation_metric_values


@MODEL_REGISTRY.register()
class DiscriminatorOfflineTrainingModel(BaseModel):
    """
    Train a discriminator offline.
    """

    def __init__(self, opt):
        super().__init__(opt)

        ###########################
        ######## Generator ########
        ###########################

        # Generator
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)
        self._load_weights(self.net_g, is_discriminator=False, requires=True)
        for parameter in self.net_g.parameters():
            parameter.requires_grad = False
        self.net_g.eval()

        ###########################
        ###### Discriminator ######
        ###########################

        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)
        self._load_weights(self.net_d, is_discriminator=True)
        self.net_d.eval()

        ###########################
        ######## Settings #########
        ###########################

        # Generator Outputs
        self.generator_output_format: dict = self.net_g.model_output_format()
        self.primary_output_key_g: str = self.net_g.primary_output()
        if opt.get('val', None):
            self.enabled_output_keys_g: List[str] = opt['val'].get('enabled_outputs', [self.primary_output_key_g])
        else:
            self.enabled_output_keys_g: List[str] = [self.primary_output_key_g]

        # Discriminator Input
        self.discriminator_input_size = self.net_d.input_size

        if self.is_train:
            self.net_d.train()
            logger = get_root_logger()
            train_opt = self.opt['train']

            # losses
            self.gan_loss = build_loss(train_opt['gan_loss']).to(self.device)
            self.grad_clip_max_norm = train_opt.get('grad_clip_max_norm', None)

            # discriminator optimizers
            self.optimizer_d = read_optimizer_options(train_opt, self.net_d, logger, is_discriminator=True)
            self.optimizers.append(self.optimizer_d)

            self.setup_schedulers()

            self.lq = None
            self.gt = None
            self.output_images = None
            self.log_dict = None

    def _load_weights(self, network, is_discriminator: bool = False, requires: bool = False):
        suffix = 'd' if is_discriminator else 'g'
        path_opt = self.opt['path']
        load_path = path_opt.get(f'pretrain_network_{suffix}', None)
        if load_path is not None:
            use_strict = path_opt.get(f'strict_load_{suffix}', True)
            param_key = path_opt.get(f'param_key_{suffix}', 'params')
            self.load_network(network, load_path, use_strict, param_key)
        elif requires:
            raise ValueError(f"Pretrained network is required for {network.__class__.__name__}")

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):

        loss_dict = OrderedDict()

        # Forward
        bundled_outputs: Dict[str, Any] = self.net_g(self.lq)
        bundled_outputs.update(gt=self.gt)

        # Backward discriminator
        l_total_d_real = 0
        l_total_d_fake = 0
        self.optimizer_d.zero_grad()
        for key in self.enabled_output_keys_g:
            sr = bundled_outputs[key]
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.gan_loss(real_d_pred, True, is_disc=True)
            loss_dict[f'out_gan_discr_real_{key}'] = torch.mean(real_d_pred.detach())
            loss_dict[f'l_gan_discr_real_{key}'] = l_d_real
            l_total_d_real += l_d_real
            # fake
            fake_d_pred = self.net_d(sr.detach())
            l_d_fake = self.gan_loss(fake_d_pred, False, is_disc=True)
            loss_dict[f'out_gan_discr_fake_{key}'] = torch.mean(fake_d_pred.detach())
            loss_dict[f'l_gan_discr_fake_{key}'] = l_d_fake
            l_total_d_fake += l_d_fake
        loss_dict[f'l_gan_discr_real'] = l_total_d_real
        loss_dict[f'l_gan_discr_fake'] = l_total_d_fake
        l_total_d_real.backward()
        l_total_d_fake.backward()

        if self.grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.grad_clip_max_norm)

        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        pbar = None
        use_pbar = self.opt['val'].get('pbar', False)
        logger = get_root_logger()

        if use_pbar:
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        self.net_d.eval()

        results_pred = {branch: [] for branch in self.enabled_output_keys_g}
        results_loss = {branch: [] for branch in self.enabled_output_keys_g}
        for idx, val_data in enumerate(dataloader):
            # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            with torch.no_grad():
                self.output_images = self._multi_branch_patch_forward_g(
                    lq=self.lq, scale=self.opt.get('scale', 1), enabled_output_keys=self.enabled_output_keys_g,
                )
                for branch, sr_image in self.output_images.items():
                    pred_values = self._forward_d(sr_image.detach(), self.discriminator_input_size)
                    loss_value = self.gan_loss(pred_values, False, is_disc=False)
                    results_pred[branch].append(torch.mean(pred_values))
                    results_loss[branch].append(loss_value)

                # tentative for out of GPU memory
                del self.gt
                del self.lq
                del self.output_images
                torch.cuda.empty_cache()

        metric_names = ['DOV', 'GGL']
        results = {branch: {metric: 0.0 for metric in metric_names} for branch in self.enabled_output_keys_g}
        for branch in self.enabled_output_keys_g:
            avg_pred_value = torch.mean(torch.stack(results_pred[branch]))
            avg_loss_value = torch.mean(torch.stack(results_loss[branch]))

            results[branch][metric_names[0]] = avg_pred_value.item()
            results[branch][metric_names[1]] = avg_loss_value.item()

        csv_file_path = osp.join(self.opt['path']['experiments_root'], 'validation_results.csv')
        log_validation_metric_values(
            metric_names=metric_names,
            current_iter=current_iter,
            dataset_name=dataset_name,
            metric_results=results,
            best_metric_results=dict(),
            primary_branch_name=self.primary_output_key_g,
            tb_logger=tb_logger,
            csv_file_path=csv_file_path,
        )

        self.net_d.train()
        if use_pbar:
            pbar.close()

    def get_current_visuals(self):
        img_lq = tensor2img(self.lq.detach().cpu())
        img_gt = tensor2img(self.gt.detach().cpu())
        img_sr = OrderedDict()
        for branch, image in self.output_images.items():
            img_sr[branch] = tensor2img(image.detach().cpu())
        images = {
            'lq': img_lq,
            'gt': img_gt,
            'sr': img_sr,
        }
        return images

    def _forward_d(self, sr: torch.Tensor, max_size: int, ) -> torch.Tensor:

        network = self.net_d

        img, col, row, mod_pad_h, mod_pad_w = calculate_and_padding_image_no_overlapping(
            sr, patch_size=max_size,
        )

        # list of partition borders
        chopping_boxes = calculate_borders_for_chopping(
            col, row, max_size, max_size, 0, 0
        )

        partitioned_img = []
        for box in chopping_boxes:
            h_range, w_range = box
            partitioned_img.append(img[..., h_range, w_range])

        partitioned_results = []
        for patch in partitioned_img:
            output = network(patch)
            partitioned_results.append(output)

        results = torch.concat(partitioned_results, dim=1)
        return results

    def _multi_branch_patch_forward_g(
            self, lq: torch.Tensor, scale: int, enabled_output_keys: list
    ) -> Dict[str, torch.Tensor]:

        network = self.net_g

        # padding image and calculate partition parameters
        img, col, row, mod_pad_h, mod_pad_w, split_h, split_w, shave_h, shave_w = calculate_and_padding_image(lq)

        # noinspection PyPep8Naming
        B, C, H, W = img.shape

        # list of partition borders
        chopping_boxes = calculate_borders_for_chopping(
            col, row, split_h, split_w, shave_h, shave_w
        )

        # list of patches / partitions
        partitioned_img = []
        for box in chopping_boxes:
            h_range, w_range = box
            partitioned_img.append(img[..., h_range, w_range])

        del chopping_boxes
        prediction_patches = {k: [] for k in enabled_output_keys}

        # image processing of each partition
        for patch in partitioned_img:
            bundled_output = network(patch)
            for key in enabled_output_keys:
                prediction_patches[key].append(bundled_output[key])
                pass

        predictions = dict()
        for key in enabled_output_keys:
            predictions[key] = recover_from_patches_and_remove_paddings(
                prediction_patches[key], col, row,
                B, C, W, H, scale,
                split_h, split_w, shave_h, shave_w,
                mod_pad_h, mod_pad_w, scale
            ).to(self.device)

        return predictions

    def save(self, epoch, current_iter):
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

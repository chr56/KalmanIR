from collections import OrderedDict
from os import path as osp
from typing import Dict, Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.binary_transform import decimal_to_binary, binary_to_decimal
from basicsr.utils.format import TableFormatter
from basicsr.utils.img_util import (
    calculate_and_padding_image,
    calculate_borders_for_chopping,
    recover_from_patches_and_remove_paddings
)
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel
from .util_common import patch_forward
from .util_config import (
    convert_format,
    MultipleLossOptions,
    read_loss_options,
    read_optimizer_options,
    valid_model_output_settings,
)


@MODEL_REGISTRY.register()
class KalmanSRModel(BaseModel):
    """
    SR model for single image super-resolution for Kalman IR.
    - has multiple image outputs
    - input and output are partitioned image patches with fixed size when validation.
    """

    def __init__(self, opt):
        super(KalmanSRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # parse network output settings
        self.model_output_format = opt['model_output']
        self.enabled_output_indexes = opt.get('model_output_enabled', None)
        self.primary_output_index = valid_model_output_settings(
            self.model_output_format, self.enabled_output_indexes
        )

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # training settings
        if self.is_train:
            self.criteria_per_output = None
            self.optimizer_g = None
            self.init_training_settings()
            self.log_gan_output = self.opt.get('log_gan_output_values', True)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        logger = get_root_logger()

        self.setup_ema_decay(train_opt, logger)

        # define losses
        losses_options, require_discriminator = read_loss_options(
            train_opt, self.device, len(self.model_output_format), logger
        )
        losses = MultipleLossOptions(losses_options, self.model_output_format, 'D')
        logger.info(f"All losses: \n{losses}")
        self.criteria_per_output = losses
        self.train_gan_discriminator = train_opt.get('train_discriminator', False)
        if require_discriminator:
            logger.info('Loading pretrained discriminator...')
            self.load_pretrained_discriminator(self.opt['network_d'])

        # set up optimizers
        self.optimizer_g = read_optimizer_options(train_opt, self.net_g, logger)
        self.optimizers.append(self.optimizer_g)

        if self.train_gan_discriminator:
            logger.info("GAN discriminator is now trainable!")
            self.train_discriminator_start_iter = train_opt.get('train_discriminator_start_iter', 0)
            self.train_discriminator_frequency = train_opt.get('train_discriminator_frequency', 1)
            self.optimizer_d = read_optimizer_options(self.opt['train'], self.net_d, logger, is_discriminator=True)
            self.optimizers.append(self.optimizer_d)

        # set up schedulers
        self.setup_schedulers()

    def load_pretrained_discriminator(self, disc_opt):
        self.net_d = build_network(disc_opt)
        self.net_d = self.model_to_device(self.net_d)
        pretrained_path = self.opt['path'].get('pretrain_network_d', None)
        if pretrained_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(
                self.net_d, pretrained_path, self.opt['path'].get('strict_load_d', True), param_key
            )
            if not self.train_gan_discriminator:
                self.net_d.eval()
                self._discriminator_parameters_grad(frozen=True)
        elif not self.train_gan_discriminator:
            raise ValueError("No pretrained discriminator network, GAN Loss not available!")

    def setup_ema_decay(self, train_opt, logger):
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):

        loss_dict = OrderedDict()

        ###
        ### optimize net_g
        ###

        if self.train_gan_discriminator:
            self._discriminator_parameters_grad(frozen=True)
            self.net_d.eval()

        self.optimizer_g.zero_grad()
        output = self.net_g(self.lq)

        l_total = self.calculate_losses(output, self.criteria_per_output.losses(), loss_dict)
        l_total.backward()

        self.optimizer_g.step()

        ###
        ### optimize net_d
        ###
        if (self.train_gan_discriminator and
                current_iter > self.train_discriminator_start_iter and
                current_iter % self.train_discriminator_frequency == 0):
            self._discriminator_parameters_grad(frozen=False)
            self.net_d.train()

            self.optimizer_d.zero_grad()
            self.backward_discriminator_losses(output, self.criteria_per_output.all_gan_losses, loss_dict)
            self.optimizer_d.step()

        ###
        ###
        ###

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def calculate_losses(self, outputs, losses, loss_dict):
        """Calculate multiple losses"""
        l_total = 0
        # calculate losses
        for output_index, losses_per_output in losses.items():
            for criterion in losses_per_output:
                name = criterion['name']
                mode = criterion['mode']
                loss_fn = criterion['loss_fn']

                gt_transform = criterion['gt_transform']
                output_transform = criterion['output_transform']

                sr = output_transform(outputs)

                if mode == 'pixel':
                    gt = gt_transform(self.gt)
                    l_pixel = loss_fn(sr, gt)
                    l_total += l_pixel
                    loss_dict[f'l_{name}'] = l_pixel
                elif mode == 'perceptual':
                    gt = gt_transform(self.gt)
                    l_perceptual, l_style = loss_fn(sr, gt)
                    if l_perceptual is not None:
                        l_total += l_perceptual
                        loss_dict[f'l_{name}' if not l_style else f'l_{name}_perceptual'] = l_perceptual
                    if l_style is not None:
                        l_total += l_style
                        loss_dict[f'l_{name}' if not l_perceptual else f'l_{name}_style'] = l_style
                elif mode == 'gan':
                    d_out = self.net_d(sr)
                    l_gan = loss_fn(d_out, True)
                    l_total += l_gan
                    loss_dict[f'l_{name}'] = l_gan
                    if self.log_gan_output:
                        loss_dict[f'out_discr_{name}'] = torch.mean(d_out.detach())
                pass
        return l_total

    def backward_discriminator_losses(self, outputs, gan_losses: list, loss_dict):

        if len(gan_losses) < 1:
            get_root_logger().warning(f'No GAN losses found, skipping discriminator loss.')
            return

        for gan_loss in gan_losses:
            name = gan_loss['name']
            loss_fn_gan = gan_loss['loss_fn']
            output_transform = gan_loss['output_transform']

            sr = output_transform(outputs)

            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = loss_fn_gan(real_d_pred, True, is_disc=True)
            loss_dict[f'l_{name}_discr_real'] = l_d_real
            loss_dict[f'out_{name}_discr_real'] = torch.mean(real_d_pred.detach())
            l_d_real.backward()
            # fake
            fake_d_pred = self.net_d(sr.detach())
            l_d_fake = loss_fn_gan(fake_d_pred, False, is_disc=True)
            loss_dict[f'l_{name}_discr_fake'] = l_d_fake
            loss_dict[f'out_{name}_discr_fake'] = torch.mean(fake_d_pred.detach())
            l_d_fake.backward()

        return loss_dict

    def forward_model(self):
        """Forward model with partitioned lq data"""
        # padding image and calculate partition parameters
        img, col, row, mod_pad_h, mod_pad_w, split_h, split_w, shave_h, shave_w = calculate_and_padding_image(self.lq)

        B, C, H, W = img.shape
        scale = self.opt.get('scale', 1)

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

        output_size = len(self.model_output_format)
        prediction_patches = {k: [] for k in range(output_size)}

        # image processing of each partition
        for patch in partitioned_img:
            raw_outputs = self.net_g(patch) if not hasattr(self, 'net_g_ema') else self.net_g_ema(patch)
            assert hasattr(raw_outputs, '__len__'), "model output must be iterable"
            assert len(raw_outputs) == output_size, "model image output size mismatched with settings"
            for k in self.enabled_output_indexes:
                patch = convert_format(
                    raw_outputs[k], from_format=self.model_output_format[k], to_format='D'
                )
                prediction_patches[k].append(patch)
                pass

        predictions = dict()
        for k in self.enabled_output_indexes:
            predictions[k] = recover_from_patches_and_remove_paddings(
                prediction_patches[k], col, row,
                B, C, W, H, scale,
                split_h, split_w, shave_h, shave_w,
                mod_pad_h, mod_pad_w, scale
            )

        self.output_images = predictions

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        metrics_opt = self.opt['val'].get('metrics')

        with_metrics = metrics_opt is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            # zero metric results
            empty_metric_dict = {metric: 0 for metric in metrics_opt.keys()}
            self.metric_results = {
                k: empty_metric_dict.copy() for k in self.enabled_output_indexes  # Output Index >> Metric Name >> Value
            }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            del empty_metric_dict

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
        else:
            self.net_g.eval()

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)

            with torch.no_grad():
                self.output_images = patch_forward(
                    network=self.net_g if not hasattr(self, 'net_g_ema') else self.net_g_ema,
                    lq=self.lq,
                    scale=self.opt.get('scale', 1),
                    output_formats=self.model_output_format,
                    output_indexes_enabled=self.enabled_output_indexes,
                )
                visuals = self.get_current_visuals()  # convert to ndarray
                img_gt = visuals['gt']
                img_sr = visuals['sr']

            # tentative for out of GPU memory
            del self.gt
            del self.lq
            del self.output_images
            torch.cuda.empty_cache()

            if save_img:
                for k, img in visuals['sr'].items():
                    save_img_path = self._saved_image_path(
                        dataset_name, img_name, current_iter, k + 1
                    )
                    imwrite(img, save_img_path)

            if with_metrics:
                # calculate metrics
                for sr_index in self.enabled_output_indexes:
                    for name, opt_ in metrics_opt.items():
                        metric_data['img'] = img_sr[sr_index]
                        metric_data['img2'] = img_gt
                        self.metric_results[sr_index][name] += calculate_metric(metric_data, opt_)
                    pass
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for k, k_metric_results in self.metric_results.items():
                for metric in k_metric_results.keys():
                    # reduce metric result
                    self.metric_results[k][metric] /= (idx + 1)
                    # update the best metric result for primary output
                    if k == self.primary_output_index:
                        self._update_best_metric_result(
                            dataset_name, metric, k_metric_results[metric], current_iter
                        )

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

        if hasattr(self, 'net_g_ema'):
            pass  # self.net_g_ema.train()
        else:
            self.net_g.train()

    def _saved_image_path(self, dataset_name, img_name, current_iter, extra_suffix) -> str:
        if self.opt['is_train']:
            file_name = f'{img_name}_{current_iter:04d}_{extra_suffix}.png'
        else:
            suffix = self.opt["val"]["suffix"] if self.opt['val']['suffix'] else self.opt["name"]
            file_name = f'{img_name}_{suffix}_{extra_suffix}.png'
        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, file_name)
        return save_img_path

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        metric_names = self.opt['val']['metrics'].keys()
        # Printable Log
        log_str = f'\n\t ### Validation {dataset_name} ###\n'
        log_str += self._printable_results(
            metric_names,
            self.best_metric_results[dataset_name] if hasattr(self, 'best_metric_results') else None,
        )
        get_root_logger().info(log_str)
        # TensorBoard
        if tb_logger:
            for k, k_metric_results in self.metric_results.items():
                for metric, value in k_metric_results.items():
                    tb_logger.add_scalar(f'metrics_{k}/{dataset_name}/{metric}', value, current_iter)
                    if self.primary_output_index == k:
                        tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def _printable_results(self, metric_names, best_metrics) -> str:
        formatter = TableFormatter(column_width=8, label_width=14, float_precision=4)
        formatter.header("# Metric", metric_names)
        for k, k_metric_results in self.metric_results.items():
            formatter.row_unordered(f"Output {k + 1}:", k_metric_results)
        if best_metrics:  # best metric is only for primary output
            best_values = []
            best_iter = []
            for metric_name in metric_names:
                best_values.append(best_metrics[metric_name]["val"])
                best_iter.append(best_metrics[metric_name]["iter"])
            formatter.new_line()
            formatter.row_ordered(f"Best Output {self.primary_output_index + 1}:", best_values)
            formatter.row_ordered(f"Best Iter {self.primary_output_index + 1}:", best_iter)
        formatter.new_line()
        result = formatter.result()
        return result

    def get_current_visuals(self):
        img_lq = tensor2img(self.lq.detach().cpu())
        img_gt = tensor2img(self.gt.detach().cpu())
        img_sr = OrderedDict()
        for idx, image in self.output_images.items():
            img_sr[idx] = tensor2img(image.detach().cpu())
        images = {
            'lq': img_lq,
            'gt': img_gt,
            'sr': img_sr,
        }
        return images

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        if hasattr(self, 'net_d'):
            self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    def _discriminator_parameters_grad(self, frozen: bool):
        for p in self.net_d.parameters():
            p.requires_grad = (not frozen)

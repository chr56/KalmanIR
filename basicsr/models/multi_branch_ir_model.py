import os.path
from collections import OrderedDict
from os import path as osp
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.losses.mixed_loss import MixedLoss
from basicsr.metrics import calculate_metric
from basicsr.utils.img_util import (
    imwrite, tensor2img,
    calculate_and_padding_image,
    calculate_borders_for_chopping,
    recover_from_patches_and_remove_paddings,
)
from basicsr.utils.logger import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.validation_util import log_validation_metric_values
from .base_model import BaseModel
from .util_config import read_optimizer_options


@MODEL_REGISTRY.register()
class MultiBranchIRModel(BaseModel):
    """SR model for single image super-resolution, supporting multiple outputs architecture."""

    def __init__(self, opt):
        super(MultiBranchIRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        # network output settings
        self.model_output_format: dict = self.net_g.model_output_format()
        self.primary_output_key: str = self.net_g.primary_output()

        # validation settings
        if opt.get('val', None):
            self.enabled_output_keys: List[str] = opt['val'].get('enabled_outputs', [self.primary_output_key])
        else:
            self.enabled_output_keys: List[str] = [self.primary_output_key]

        logger = get_root_logger()
        if self.is_train:
            self.net_g.train()
            train_opt = self.opt['train']

            # set up EMA decay
            self.ema_decay = train_opt.get('ema_decay', 0)
            if self.ema_decay > 0:
                self._setup_ema_decay()

            # set up lose functions
            if train_opt.get('mixed_losses', None):
                loss = build_loss(train_opt['mixed_losses']).to(self.device)
                assert isinstance(loss, MixedLoss), f"'mixed_losses' must be a MixedLoss object, got {type(loss)}"
                loss.validate(self.model_output_format.keys())
                logger.info('\n' + loss.summary())
                self.loss_fn = loss
            else:
                raise NotImplementedError(f"Support mixed losses only!")

            # set up optimizers
            self.optimizer_g = read_optimizer_options(train_opt, self.net_g, logger)
            self.optimizers.append(self.optimizer_g)
            # set up LR schedulers
            self.setup_schedulers()

            # set up gradient clipping
            self.grad_clip_max_norm = train_opt.get('grad_clip_max_norm', None)

            # frozen parameters
            target_frozen_parameters_groups: list[str] = train_opt.get('frozen_parameters', [])
            if target_frozen_parameters_groups and hasattr(self.net_g, 'partitioned_parameters'):
                existed_parameters = self.net_g.partitioned_parameters()
                for target in target_frozen_parameters_groups:
                    params = existed_parameters.get(target, None)
                    if params is not None:
                        logger.info(f"Frozen parameters group {target} ...")
                        for p in params:
                            p.requires_grad = False
                    else:
                        raise ValueError(f"Parameters group {target} does not exist! ")
                pass
            pass

        self.log_dict = None
        self.lq = None
        self.gt = None
        self.output_images = None
        self.metric_results: Dict[str, Dict[str, Any]] = dict()

    def _setup_ema_decay(self):
        logger = get_root_logger()
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

        self.optimizer_g.zero_grad()
        bundled_outputs: Dict[str, Any] = self.net_g(self.lq)
        bundled_outputs.update(gt=self.gt)

        l_total, loss_dict = self.loss_fn(bundled_outputs)

        l_total.backward()
        if self.grad_clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=self.grad_clip_max_norm)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):

        dataset_name: str = dataloader.dataset.opt['name']
        use_pbar: bool = self.opt['val'].get('pbar', False)
        metrics_opt: Dict[str, Any] = self.opt['val'].get('metrics')
        with_metrics = metrics_opt is not None
        metric_names = metrics_opt.keys()

        if with_metrics:
            # zero metric results
            empty_metric_dict = {metric: 0.0 for metric in metric_names}
            self.metric_results = {
                k: empty_metric_dict.copy() for k in self.enabled_output_keys  # Output Index >> Metric Name >> Value
            }
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
            del empty_metric_dict

        metric_data = dict()
        pbar = None

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
                self.output_images = self._multi_branch_patch_forward(
                    lq=self.lq,
                    scale=self.opt.get('scale', 1),
                    enabled_output_keys=self.enabled_output_keys,
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
                for branch, img in visuals['sr'].items():
                    save_img_path = self._saved_image_path(
                        dataset_name, img_name, current_iter, branch + 1
                    )
                    imwrite(img, save_img_path)

            if with_metrics:
                # calculate metrics
                for key in self.enabled_output_keys:
                    for name, opt_ in metrics_opt.items():
                        metric_data['img'] = img_sr[key]
                        metric_data['img2'] = img_gt
                        self.metric_results[key][name] += calculate_metric(metric_data, opt_)
                    pass
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if with_metrics:
            for branch, branch_metric_results in self.metric_results.items():
                for metric in branch_metric_results.keys():
                    # reduce metric result
                    self.metric_results[branch][metric] /= (idx + 1)
                    # update the best metric result for primary output
                    if branch == self.primary_output_key:
                        self._update_best_metric_result(
                            dataset_name, metric, branch_metric_results[metric], current_iter
                        )
            csv_file_path = os.path.join(self.opt['path']['log'], 'validation_results.csv')
            log_validation_metric_values(
                metric_names=metric_names,
                current_iter=current_iter,
                dataset_name=dataset_name,
                metric_results=self.metric_results,
                best_metric_results=self.best_metric_results[dataset_name],
                primary_branch_name=self.primary_output_key,
                tb_logger=tb_logger,
                csv_file_path=csv_file_path,
            )

        if use_pbar:
            pbar.close()

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
        # noinspection PyTypeChecker
        return save_img_path

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

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)

    def _multi_branch_patch_forward(
            self, lq: torch.Tensor, scale: int, enabled_output_keys: list
    ) -> Dict[str, torch.Tensor]:

        network = self.net_g if not hasattr(self, 'net_g_ema') else self.net_g_ema

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
            )

        return predictions

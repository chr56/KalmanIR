from collections import OrderedDict

import torch

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.models.util_common import patch_forward
from basicsr.models.util_config import (
    config_suffix,
    convert_format,
    valid_model_output_settings,
    frozen_model_parameters,
    read_optimizer_options,
)
from basicsr.utils.logger import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register()
class DiscriminatorTrainerModule(BaseModel):
    """
    Model to train a discriminator independently.
    """

    def __init__(self, opt):
        super().__init__(opt)

        ###########################

        # define Generator
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load Generator and froze
        self._load_weights(self.net_g, is_discriminator=False, requires=True)
        self.net_g.eval()

        frozen_model_parameters(self.net_g, frozen=True)

        ###########################

        # define Discriminator
        self.net_d = build_network(opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load Generator
        self._load_weights(self.net_d, is_discriminator=True)
        self.net_d.eval()

        ###############################

        # Generator settings
        self.generator_output_format = opt['model_output']
        self.enabled_generator_output_indexes = opt.get('model_output_enabled', None)
        valid_model_output_settings(
            self.generator_output_format,
            self.enabled_generator_output_indexes,
        )

        # Discriminator settings
        self.log_discriminator_output = self.opt.get('log_discriminator_output_values', True)

        if self.is_train:
            self.net_d.train()
            logger = get_root_logger()
            train_opt = self.opt['train']

            # losses
            self.gan_loss = build_loss(train_opt['gan_loss']).to(self.device)

            # discriminator optimizers
            self.optimizer_d = read_optimizer_options(train_opt, self.net_d, logger, is_discriminator=True)
            self.optimizers.append(self.optimizer_d)

            self.setup_schedulers()

            self.lq = None
            self.gt = None
            self.log_dict = None

        else:
            self.net_d.eval()

    def _load_weights(self, network, is_discriminator: bool = False, requires: bool = True):
        suffix = config_suffix(is_discriminator)
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
        outputs = self.net_g(self.lq)

        # Backward discriminator
        l_total_d_real = 0
        l_total_d_fake = 0
        self.optimizer_d.zero_grad()
        for k in self.enabled_generator_output_indexes:
            sr = convert_format(outputs[k], self.generator_output_format[k], 'D')
            # real
            real_d_pred = self.net_d(self.gt)
            l_d_real = self.gan_loss(real_d_pred, True, is_disc=True)
            loss_dict[f'out_gan_discr_real_{k}'] = torch.mean(real_d_pred.detach())
            loss_dict[f'l_gan_discr_real_{k}'] = l_d_real
            l_total_d_real += l_d_real
            # fake
            fake_d_pred = self.net_d(sr.detach())
            l_d_fake = self.gan_loss(fake_d_pred, False, is_disc=True)
            loss_dict[f'out_gan_discr_fake_{k}'] = torch.mean(fake_d_pred.detach())
            loss_dict[f'l_gan_discr_fake_{k}'] = l_d_fake
            l_total_d_fake += l_d_fake
        loss_dict[f'l_gan_discr_real'] = l_total_d_real
        loss_dict[f'l_gan_discr_fake'] = l_total_d_fake
        l_total_d_real.backward()
        l_total_d_fake.backward()
        self.optimizer_d.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        pbar = None
        use_pbar = self.opt['val'].get('pbar', False)

        if use_pbar:
            from tqdm import tqdm
            pbar = tqdm(total=len(dataloader), unit='image')

        self.net_d.eval()

        # for idx, val_data in enumerate(dataloader):
        #     # img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
        #     self.feed_data(val_data)
        #
        #     with torch.no_grad():
        #         values = {k: [] for k in self.enabled_generator_output_indexes}
        #         predictions = patch_forward(
        #             network=self.net_g,
        #             lq=self.lq,
        #             scale=self.opt.get('scale', 1),
        #             output_formats=self.generator_output_format,
        #             output_indexes_enabled=self.enabled_generator_output_indexes,
        #         )
        #         for k, sr_image in predictions.items():
        #             pred_value = self.net_d(sr_image)
        #             values[k].append(pred_value)
        #
        #         # tentative for out of GPU memory
        #         del self.gt
        #         del self.lq
        #         del predictions
        #         torch.cuda.empty_cache()
        #
        #         for k, values in values.items():
        #             avg_value = torch.mean(*values)
        #             tb_logger.add_scalar(f'metrics_gan/{dataset_name}/out_gan_{k}', avg_value, current_iter)
        #
        #     pass

        self.net_d.train()
        if use_pbar:
            pbar.close()

    def save(self, epoch, current_iter):
        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

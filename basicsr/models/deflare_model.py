from collections import OrderedDict
from os import path as osp
import numpy as np
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.flare_util import predict_flare_from_6_channel,predict_flare_from_3_channel
from basicsr.metrics import calculate_metric
import torch
from tqdm import tqdm
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import cv2
from torchvision.transforms import Compose
from torch import Tensor
from sklearn.preprocessing import MinMaxScaler

@MODEL_REGISTRY.register()
class DeflareModel(SRModel):

    def init_training_settings(self):
        self.accumulated_iter = 2 # simulate desired batch size
        self.currently_validating = False
        self.net_g.train()
        train_opt = self.opt['train']
        self.output_ch=self.opt['network_g']['output_ch']
        if 'multi_stage' in self.opt['network_g']:
            self.multi_stage=self.opt['network_g']['multi_stage']
        else:
            self.multi_stage=1
        print("Output channel is:", self.output_ch)
        print("Network contains",self.multi_stage,"stages.")

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
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

        # define losses
        self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)
        self.l_perceptual = build_loss(train_opt['perceptual']).to(self.device)

        # Define depth model
        net_w, net_h = 384, 384
        self.depth_model = DPTDepthModel(
            path="DPT/dpt_hybrid-midas-501f0c75.pt",
            backbone="vitb_rn50_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        self.depth_normalize = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        self.depth_transform = Compose(
                [
                    Resize(
                        net_w,
                        net_h,
                        resize_target=None,
                        keep_aspect_ratio=True,
                        ensure_multiple_of=32,
                        resize_method="minimal",
                        image_interpolation_method=cv2.INTER_CUBIC,
                    ),
                    self.depth_normalize,
                    PrepareForNet(),
                ]
            )

        self.depth_model.eval()
        self.depth_model.to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.gt = data['gt'].to(self.device)
        self.input_for_loss = data['lq'].to(self.device)
        data['lq'] = Tensor(data['lq']).cpu()

        if 'flare' in data:
            self.flare = data['flare'].to(self.device)
            self.gamma = data['gamma'].to(self.device)
        if 'mask' in data:
            self.mask = data['mask'].to(self.device)

        img = np.uint8(data['lq'].permute(0, 2, 3, 1).numpy()*255)
        dpt_depth = self.predict_depth_dpt(img)
        self.lq = torch.from_numpy(np.concatenate((data['lq'], dpt_depth.reshape((1, 1, 512, 512))), axis=1)).to(self.device)

    def predict_depth_dpt(self, img):
        img_input = np.zeros((img.shape[0], 3, 384, 384), dtype=np.float32)
        for i in range(img.shape[0]):
            img_input[i] = self.depth_transform({"image": img[i]})["image"]
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(self.device)

            prediction = self.depth_model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=512,
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )
        out = np.zeros(prediction.shape, dtype=prediction.dtype)
        for i in range(img.shape[0]):
            depth_min = prediction.min()
            depth_max = prediction.max()
            # Clip prediction to [0, 3000]
            prediction = np.clip(prediction, 0, 3000)
            if depth_max - depth_min > np.finfo("float").eps:
                out =  prediction/3000 #65535 * (prediction- 0) / (depth_max - depth_min)
        return out

    def optimize_parameters(self, current_iter):
        
        self.output = self.net_g(self.lq)
        
        if self.output_ch==6:
            self.deflare,self.flare_hat,self.merge_hat=predict_flare_from_6_channel(self.output,self.gamma)
        elif self.output_ch==3:
            self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            self.deflare,self.flare_hat=predict_flare_from_3_channel(self.output,self.mask,self.lq,self.flare,self.lq,self.gamma)        
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        if  not hasattr(self, 'loss_dict')  or (current_iter) % self.accumulated_iter == 0:
            self.loss_dict = OrderedDict()
            self.loss_dict['l1_recons'] = self.loss_dict['l1_flare'] = self.loss_dict['l1_base'] = self.loss_dict['l1'] = 0
            self.loss_dict['l_vgg'] = self.loss_dict['l_vgg_flare'] = self.loss_dict['l_vgg_base'] = 0
            self.loss_dict['l_total'] = 0

        # l1 loss
        l_total = 0
        l1_flare = self.l1_pix(self.flare_hat, self.flare)/self.accumulated_iter
        l1_base = self.l1_pix(self.deflare, self.gt)/self.accumulated_iter
        l1 = l1_flare+l1_base
        if self.output_ch==6:
            l1_recons= 2*self.l1_pix(self.merge_hat, self.input_for_loss)/self.accumulated_iter
            self.loss_dict['l1_recons'] +=l1_recons
            l1+=l1_recons
        l_total += l1
        self.loss_dict['l1_flare'] += l1_flare
        self.loss_dict['l1_base'] += l1_base
        self.loss_dict['l1'] += l1

        # perceptual loss
        l_vgg_flare = self.l_perceptual(self.flare_hat, self.flare)/self.accumulated_iter
        l_vgg_base = self.l_perceptual(self.deflare, self.gt)/self.accumulated_iter
        l_vgg = l_vgg_base +l_vgg_flare
        l_total += l_vgg
        self.loss_dict['l_vgg'] += l_vgg
        self.loss_dict['l_vgg_base'] += l_vgg_base
        self.loss_dict['l_vgg_flare'] += l_vgg_flare
        self.loss_dict['l_total'] += l_total
        l_total.backward()

        if (current_iter) % self.accumulated_iter == 0:
            self.optimizer_g.step()
            self.optimizer_g.zero_grad()
            self.log_dict = self.reduce_loss_dict(self.loss_dict)
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
        if self.output_ch==6:
            self.deflare,self.flare_hat,self.merge_hat=predict_flare_from_6_channel(self.output,self.gamma)
        elif self.output_ch==3:
            self.mask=torch.zeros_like(self.lq).cuda() # Comment this line if you want to use the mask
            self.deflare,self.flare_hat=predict_flare_from_3_channel(self.output,self.mask,self.gt,self.flare,self.lq,self.gamma)        
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."
        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.currently_validating = True
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)
        self.currently_validating = False
        

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        self.currently_validating = True
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()
            img_name='deflare_'+str(idx).zfill(5)+'_'
            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
        self.currently_validating = False

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.input_for_loss.detach().cpu()
        #self.blend= blend_light_source(self.lq[:,:-1,:,:], self.deflare, 0.97)
        out_dict['result']= self.deflare.detach().cpu() # self.blend.detach().cpu()
        out_dict['flare']=self.flare_hat.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict
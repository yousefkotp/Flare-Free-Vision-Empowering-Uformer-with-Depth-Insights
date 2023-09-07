import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from basicsr.archs.uformer_arch import Uformer
import argparse
from basicsr.archs.unet_arch import U_Net
from basicsr.utils.flare_util import mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
import torchvision.transforms as transforms
import os
import gc
from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
import cv2
from torchvision.transforms import Compose


parser = argparse.ArgumentParser()
parser.add_argument('--input',type=str,default=None)
parser.add_argument('--output',type=str,default=None)
parser.add_argument('--model_type',type=str,default='Uformer')
parser.add_argument('--model_path',type=str,default='checkpoint/flare7kpp/net_g_last.pth')
parser.add_argument('--output_ch',type=int,default=6)
parser.add_argument('--flare7kpp', action='store_const', const=True, default=False) #use flare7kpp's inference method and output the light source directly.

args = parser.parse_args()
model_type=args.model_type
images_path=os.path.join(args.input,"*.*")
result_path=args.output
pretrain_dir=args.model_path
output_ch=args.output_ch

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:
		os.makedirs(path)

def load_params(model_path):
     full_model=torch.load(model_path)
     if 'params_ema' in full_model:
          return full_model['params_ema']
     elif 'params' in full_model:
          return full_model['params']
     else:
          return full_model

def demo(images_path,output_path,model_type,output_ch,pretrain_dir,flare7kpp_flag):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path=glob.glob(images_path)
    result_path=output_path
    torch.cuda.empty_cache()
    if model_type=='Uformer':
        model=Uformer(img_size=512,img_ch=4,output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    elif model_type=='U_Net' or model_type=='U-Net':
        model=U_Net(img_ch=3,output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
    else:
        assert False, "This model is not supported!!"

    net_w, net_h = 384, 384
    depth_model = DPTDepthModel(
        path="DPT/dpt_hybrid-midas-501f0c75.pt",
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    depth_normalize = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    depth_transform = Compose(
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
                depth_normalize,
                PrepareForNet(),
            ]
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth_model.eval()
    depth_model.to(device)
    model.eval()
    
    for i,image_path in tqdm(enumerate(test_path)):
        image_name = image_path.split("\\")[-1]
        image_name = image_name[:6]
        if not flare7kpp_flag:
            mkdir(result_path+"deflare/")
            deflare_path = result_path+"deflare/"+str(i).zfill(5)+"_deflare.png"

        mkdir(result_path+"flare/")
        mkdir(result_path+"input/")
        mkdir(result_path+"blend/")
        
        flare_path = result_path+"flare/"+image_name+"_flare.png"
        merge_path = result_path+"input/"+image_name+"_input.png"
        blend_path = result_path+"blend/"+image_name+"_blend.png"


        img = Image.open(image_path).convert('RGB')
        img = transforms.ToTensor()(img).unsqueeze(0)
        original_img = img.clone()
        img = np.uint8(img.permute(0,2,3,1).numpy()*255)
        img_input = np.zeros((1, 3, 384, 384), dtype=np.float32)
        # For each image in the batch, we predict the depth map
        for i in range(img.shape[0]):
            img_input[i] = depth_transform({"image": img[i]})["image"]
        
        with torch.no_grad():
            sample = torch.from_numpy(img_input).to(device)

            prediction = depth_model.forward(sample)
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
            prediction = np.clip(prediction, 0, 3000)
            if depth_max - depth_min > np.finfo("float").eps:
                out =  prediction/3000 #65535 * (prediction- 0) / (depth_max - depth_min)
        merge_img = torch.from_numpy(np.concatenate((original_img, out.reshape((1, 1, 512, 512))), axis=1)).to(device)
        with torch.no_grad():
            output_img=model(merge_img).cuda()
            #if ch is 6, first three channels are deflare image, others are flare image
            #if ch is 3, unsaturated region of output is the deflare image.
            gamma=torch.Tensor([2.2])
            if output_ch==6:
                deflare_img,flare_img_predicted,merge_img_predicted=predict_flare_from_6_channel(output_img,gamma)
            elif output_ch==3:
                flare_mask=torch.zeros_like(merge_img)
                deflare_img,flare_img_predicted=predict_flare_from_3_channel(output_img,flare_mask,output_img,merge_img,merge_img,gamma)
            else:
                assert False, "This output_ch is not supported!!"
            
            torchvision.utils.save_image(original_img, merge_path)
            torchvision.utils.save_image(flare_img_predicted, flare_path)
            if flare7kpp_flag:
                    torchvision.utils.save_image(deflare_img, blend_path) # deflare_img
            else:
                #blend_img= blend_light_source(original_merge_img, deflare_img, 0.97)
                torchvision.utils.save_image(deflare_img, deflare_path)
                #torchvision.utils.save_image(blend_img, blend_path)

       
demo(images_path,result_path,model_type,output_ch,pretrain_dir,args.flare7kpp)
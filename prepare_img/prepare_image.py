import fire
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
import timm
from pathlib import Path
from PIL import Image
import copy
from utils.sketch_utils import _transform,get_sketch
from utils.utils import VIT_MODEL, set_requires_grad, SKETCH_PER_VIEW, ensure_directory
from utils.shapenet_utils import snc_synth_id_to_category_all, snc_synth_id_to_category_5, snc_category_to_synth_id_all
from utils.mesh_utils import process_sdf, augment_sdf

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from pytorch_lightning.plugins import DDPPlugin
import open_clip
from tqdm import tqdm

import argparse



class our_Dataset(torch.utils.data.Dataset):
    def __init__(self, images_folder: str,  out_dir="", data_class: str = "all",
                 image_resolution: int = 224, start_index: int = 0, end_index: int = 10000, feature_extractor_flag = 'clip', preprocess=None):
        super().__init__()
        if preprocess is None:
            self.preprocess = _transform(image_resolution)
        else:
            self.preprocess = preprocess
        self.feature_extractor_flag = feature_extractor_flag
        self.out_dir = out_dir
        self.mixed_data = False
        if data_class != "all":
            self.images_paths = []
            self.save_path = []
            self.exists_lst = []
            for each_data_id in data_class:
                sub_path = os.path.join(images_folder, each_data_id)
                for root, dirs, files in os.walk(sub_path):
                    for file in files:
                        if file.endswith('png'):
                            img_path = os.path.join(root, file)
                            save = root[:root.rfind('/')].replace(images_folder, self.out_dir)
                            save_file = os.path.join(save, file.replace('png', 'npy'))
                            if os.path.exists(save_file):
                                self.exists_lst.append(img_path)
                                # print(img_path)
                                continue
                            self.images_paths.append(img_path)
                            self.save_path.append(save_file)

            self.images_paths = self.images_paths[start_index:end_index]
            self.save_path = self.save_path[start_index:end_index]
        elif data_class == "all":
            self.images_paths = []
            self.save_path = []
            self.exists_lst = []
            for root, dirs, files in os.walk(images_folder):
                for file in files:
                    if file.endswith('png'):
                        img_path = os.path.join(root, file)
                        save = root[:root.rfind('/')].replace(images_folder, self.out_dir)
                        save_file = os.path.join(save, file.replace('png', 'npy'))
                        if os.path.exists(save_file):
                            self.exists_lst.append(img_path)
                            # print(img_path)
                            continue
                        self.images_paths.append(img_path)
                        self.save_path.append(save_file)
                        
            self.images_paths = self.images_paths[start_index:end_index]
            self.save_path = self.save_path[start_index:end_index]
        else:
            raise NotImplementedError

        print(f"need to deal number: {len(self.images_paths)}")
        print(self.images_paths)
        print(f"exists number: {len(self.exists_lst)}")

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, index):
        image_path = self.images_paths[index]
        image_path = str(image_path)
        img = self.preprocess(Image.open(image_path))
        save_path = self.save_path[index]

        return {'images': img,
                'foo': np.random.rand(10).astype(np.float32),
                'save_path': save_path}


class GenerationDataModel(LightningModule):
    def __init__(
        self,
        images_folder: str = "",
        data_class: str = "chair",
        results_folder: str = './results',
        start_index: int = 0,
        end_index: int = -1,
        feature_extractor_flag: str = "clip", # or vit\clip
        batch_size: int = 64
    ):
        super().__init__()
        self.num_workers = os.cpu_count()
        self.images_folder = images_folder
        self.data_class = data_class
        self.out_dir = results_folder
        self.feature_extractor_flag = feature_extractor_flag
        if feature_extractor_flag == "vit":
            print("------Success load VIT model---------")
            self.feature_extractor = timm.create_model(VIT_MODEL, pretrained=True)
            self.preprocess = None
        elif feature_extractor_flag == "clip":
            print("------Success load CLIP model---------")
            self.feature_extractor, _, self.preprocess  = open_clip.create_model_and_transforms("EVA02-E-14-plus", pretrained='pretrain_model/open_clip_pytorch_model.bin') 
        self.model = nn.Linear(10, 1)
        set_requires_grad(self.feature_extractor, False)
        self.start_index = start_index
        self.end_index = end_index
        self.batch_size = batch_size

    def train_dataloader(self):
        _dataset = our_Dataset(images_folder=self.images_folder,
                               out_dir=self.out_dir, data_class=self.data_class,
                               start_index=self.start_index, end_index=self.end_index, 
                               preprocess=self.preprocess, feature_extractor_flag=self.feature_extractor_flag)
        dataloader = DataLoader(_dataset,
                                num_workers=self.num_workers,
                                batch_size=self.batch_size, shuffle=False, pin_memory=True, drop_last=False)
        return dataloader

    def training_step(self, batch, index):
        foo_data = batch['foo']
        images = batch['images']
        save_path = batch['save_path']
        loss = self.model(foo_data).mean()

        with torch.no_grad():
            if self.feature_extractor_flag == 'vit':
                image_features = self.feature_extractor.forward_features(
                    images).squeeze().cpu().numpy()
            elif self.feature_extractor_flag == 'clip':
                image_features = self.feature_extractor.encode_image(images)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_features = image_features.squeeze().cpu().numpy()
            for i in tqdm(range(len(image_features)), desc="Saving Features", unit="sample"):
                ensure_directory(save_path[i][:save_path[i].rfind('/')])
                np.save(save_path[i], image_features[i])
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=1e-4)
        return None

#['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010']
#['0011','0012','0013','0014','0015','0016','0017','0018','0019','0020']
#['0021','0022','0023','0024','0025','0026','0027','0028','0029','0030']
#['0031','0032','0033','0034','0035','0036','0037','0038','0039','0040']
#['0041','0042','0043','0044','0045','0046','0047','0048','0049','0050']
#['0051','0052','0053','0054','0055','0056','0057','0058','0059','0060']
#['0061','0062','0063','0064','0065','0066','0067','0068','0069','0070']
#['0071','0072','0073','0074','0075','0076','0077','0078','0079','0080']
#['0081','0082','0083','0084','0085','0086','0087','0088','0089','0090']
#['0091','0092','0093','0094','0095','0096','0097','0098','0099','0100']
#['0101','0102','0103','0104','0105','0106','0107','0108']

def train_from_folder(
    images_folder: str,
    results_folder: str,
    # data_class=['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009','0010'],   
    data_class:list,      #"all",
    start_index: int ,
    end_index: int ,
    batch_size: int
):
    print(f"Need to deal id: {data_class}")
    if data_class == "mixed":
        seed_everything(start_index)
    model_args = dict(
        results_folder=results_folder,
        images_folder=images_folder,
        data_class=data_class,
        start_index=start_index,
        end_index=end_index,
        batch_size=batch_size
    )
    model = GenerationDataModel(**model_args)

    trainer = Trainer(devices=-1,
                      accelerator="gpu",
                      strategy=DDPPlugin(
                          find_unused_parameters=False),
                      max_epochs=1,
                      log_every_n_steps=1,)

    trainer.fit(model)

    print(f"Success: {data_class}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./input", required=True, help="input folder path")
    parser.add_argument("--output", default="./output", required=True, help="output folder path")
    parser.add_argument("--data_class", type=list,default=['0000','0001','0002'],required=True,  help="floder name")
    parser.add_argument("--start_index",type=int, default=0, required=True, help="start index")
    parser.add_argument("--end_index", type=int, default=-1,  required=True, help="end index")
    parser.add_argument("--batch_size", type=int, default=64, required=True, help="batch size")
    args = parser.parse_args()

    train_from_folder(images_folder=args.input, 
                  results_folder=args.output, 
                  data_class=args.data_class,  
                  start_index=args.start_index, 
                  end_index=args.end_index, 
                  batch_size=args.batch_size)
# python xxxxxxx.py --input xxxxxx --output xxxxxxx

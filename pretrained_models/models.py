import torch
import sys
sys.path.append(r"c:\Users\aarus\everything_dsg\adversarial-patch-transferability")
# from pretrained_models.ICNet.icnet import ICNet
# from pretrained_models.BisNetV1.model import BiSeNetV1
# from pretrained_models.BisNetV2.model import BiSeNetV2
from pretrained_models.PIDNet.model import PIDNet, get_pred_model
# from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import torch.nn.functional as F

class Models():
  def __init__(self,config):
    self.config = config
    self.name = config.model.name
    self.device = config.experiment.device
    self.model = None

  def get(self):
    if 'pidnet' in self.config.model.name:
      if '_s' in self.config.model.name:
        model = torch.load(r'C:\Users\aarus\everything_dsg\adversarial-patch-transferability\pretrained_models\PIDNet\PIDNet_S_Cityscapes_test.pt',map_location=self.device)
      if '_m' in self.config.model.name:
        model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/PIDNet/PIDNet_M_Cityscapes_test.pt',map_location=self.device)
      if '_l' in self.config.model.name:
        model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/PIDNet/PIDNet_L_Cityscapes_test.pt',map_location=self.device)
      
  
      pidnet = get_pred_model(name = self.config.model.name, num_classes = 19).to(self.device)
      if 'state_dict' in model:
          model = model['state_dict']
      model_dict = pidnet.state_dict()
      model = {k[6:]: v for k, v in model.items() # k[6:] to start after model. in key names
                          if k[6:] in model_dict.keys()}

      pidnet.load_state_dict(model)
      self.model = pidnet
      self.model.eval()
      

    # if 'bisenet' in self.config.model.name:
    #   if '_v1' in self.config.model.name:
    #     model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/BisNetV1/bisnetv1.pth',map_location=self.device)
    #     bisenet = BiSeNetV1(19,aux_mode = 'eval').to(self.device)
    #     bisenet.load_state_dict(model, strict=False)
    #   if '_v2' in self.config.model.name:
    #     model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/BisNetV2/bisnetv2.pth',map_location=self.device)
    #     bisenet = BiSeNetV2(19,aux_mode = 'eval').to(self.device)
    #     bisenet.load_state_dict(model, strict=False)
    #   self.model = bisenet
    #   self.model.eval()


    # if 'icnet' in self.config.model.name:
    #   model = torch.load('/content/drive/MyDrive/Colab Notebooks/1_Papers/3_Attack_generation/pretrained_models/ICNet/Copy of resnet50_2024-12-22 08:52:50 EST-0500_176_0.661.pth.tar',map_location=self.device)
    #   icnet = ICNet(nclass = 19).to(self.device)
    #   icnet.load_state_dict(model['model_state_dict'])
    #   self.model = icnet
    #   self.model.eval()

    # if 'segformer' in self.config.model.name:
    #   feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    #   segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to(self.device)
    #   self.model = segformer
    #   self.model.eval()


  def predict(self,image_standard,size):
    image_standard = image_standard.to(self.device)
    outputs = self.model(image_standard)
    if 'pidnet' in self.config.model.name:
      output = F.interpolate(
                    outputs[self.config.test.output_index_pidnet], size[-2:],
                    mode='bilinear', align_corners=True
                )

    # if 'segformer' in self.config.model.name:
    #   output = F.interpolate(
    #                 outputs.logits, size[-2:],
    #                 mode='bilinear', align_corners=True
    #             )

    # if 'icnet' in self.config.model.name:
    #   output = outputs[self.config.test.output_index_icnet]

    # if 'bisenet' in self.config.model.name:
    #   ## Images needs to be unnormalized and then normalized as:
    #   ## mean=[0.3257, 0.3690, 0.3223], std=[0.2112, 0.2148, 0.2115]
    #   ## The it will give 75% miou instead of 71 and to keep things simple keeping it as it
    #   output = outputs[self.config.test.output_index_bisenet]

    return output

    






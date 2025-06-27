# Unifying Biomedical Vision-Language Expertise: Towards a Generalist Foundation Model via Multi-CLIP Knowledge Distillation


This repository contains the inference code for MMKD-CLIP, a visual language model distilled from 9 well-known biomedical CLIPs. We will publish the full training pipeline when the paper is accepted.


> **<p align="justify"> Abstract:** *CLIP models pretrained on natural images with billion-scale image-text pairs have demonstrated impressive capabilities in zero-shot classification, cross-modal retrieval, and open-ended visual answering. However, transferring this success to biomedicine is hindered by the scarcity of large-scale biomedical image-text corpora, the heterogeneity of image modalities, and fragmented data standards across institutions. These limitations hinder the development of a unified and generalizable biomedical foundation model trained from scratch. To overcome this, we introduce MMKD-CLIP, a generalist biomedical foundation model developed via Multiple Medical CLIP Knowledge Distillation. Rather than relying on billion-scale raw data, MMKD-CLIP distills knowledge from nine state-of-the-art domain-specific or generalist biomedical CLIP models, each pretrained on millions of biomedical image-text pairs. Our two-stage training pipeline first performs CLIP-style pretraining on over 2.9 million biomedical image-text pairs from 26 image modalities, followed by feature-level distillation using over 19.2 million feature pairs extracted from teacher models. We evaluate MMKD-CLIP on 58 diverse biomedical datasets, encompassing over 10.8 million biomedical images across nine image modalities. The evaluation spans six core task types: zero-shot classification, linear probing, cross-modal retrieval, visual question answering, survival prediction, and cancer diagnosis. MMKD-CLIP consistently outperforms all teacher models while demonstrating remarkable robustness and generalization across image domains and task settings. These results underscore that multi-teacher knowledge distillation is a scalable and effective paradigm for building high-performing biomedical foundation models under the practical constraints of real-world data availability.* </p>

## Quick Start for inference with UniMed-CLIP models 

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from open_clip import create_model_and_transforms, get_mean_std, HFTokenizer
from PIL import Image
import torch

mean, std = get_mean_std()
model, _, preprocess = create_model_and_transforms(
    model_name='ViT-B-16-quickgelu',
    pretrained="./MMKD_B16.pth",
    precision='amp',
    device='cuda:0',
    force_quick_gelu=True,
    mean=mean, std=std,
    inmem=True,
    text_encoder_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

tokenizer = HFTokenizer(
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    context_length=256,
    **{},)

# Prepare text prompts using different class names
text_prompts = [
                  "This X-ray shows no signs of disease in the thoracic cavity.",
                  "Fundus image showing no signs of diabetic retinopathy.",
                  "Histopathology slide of Colon adenocarcinoma with glandular structures.",
                  "Endoscopic findings consistent with reflux esophagitis.",
                  "CT scan showing space-occupying lesion within the brain parenchyma.",
                  "MRI reveals contrast-enhancing brain tumor with mass effect.",
                  "Ultrasound image of the breast showing irregular margins and malignant features.",
                  "Benign mole with uniform color and symmetry indicating melanocytic nevus.",
                  "A retinal OCT image of Age-related Macular Degeneration.",
                ]
texts = [tokenizer(cls_text).to(next(model.parameters()).device, non_blocking=True) for cls_text in text_prompts]
texts = torch.cat(texts, dim=0)

# Load and preprocess images
test_imgs = [
    './imgs/xray.png',
    './imgs/fundus.jpeg',
    './imgs/pathology.jpeg',
    './imgs/endoscopy.jpg',
    './imgs/ct.jpg',
    './imgs/mri.jpg',
    './imgs/ultrasound.png',
    './imgs/dermatology.jpeg',
    './imgs/oct.jpg',
]
images = torch.stack([preprocess(Image.open(img)) for img in test_imgs]).to("cuda:0")

with torch.no_grad():
    text_features = model.encode_text(texts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    image_features = model.encode_image(images)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    logits = (model.logit_scale.exp() * image_features @ text_features.t()).detach().softmax(dim=-1)
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)

    logits = logits.cpu().numpy()


top_k = -1
for i, img in enumerate(test_imgs):
    pred = text_prompts[sorted_indices[i][0]]

    top_k = len(text_prompts) if top_k == -1 else top_k
    print(img.split('/')[-1] + ':')
    for j in range(top_k):
        jth_index = sorted_indices[i][j]
        print(f'{text_prompts[jth_index]}: {logits[i][jth_index]}')
    print('\n')
```

<details>
<summary>Outputs</summary>
Note: The addition of the image feature normalization line in the demo code could cause slight fluctuations in the probabilities. However, the arg-max of predictions (rankings) remains the same. 

```python
xray.png:
This X-ray shows no signs of disease in the thoracic cavity.: 1.0
Fundus image showing no signs of diabetic retinopathy.: 1.2106992741678368e-08
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 1.0082889190243804e-08
Ultrasound image of the breast showing irregular margins and malignant features.: 3.2723782017463066e-10
A retinal OCT image of Age-related Macular Degeneration.: 5.201791146262902e-13
Histopathology slide of Colon adenocarcinoma with glandular structures.: 1.8598925839111852e-13
Endoscopic findings consistent with reflux esophagitis.: 1.0586958928613093e-14
CT scan showing space-occupying lesion within the brain parenchyma.: 2.771852639451601e-15
MRI reveals contrast-enhancing brain tumor with mass effect.: 6.992315874843863e-17


fundus.jpeg:
Fundus image showing no signs of diabetic retinopathy.: 0.9999988079071045
A retinal OCT image of Age-related Macular Degeneration.: 1.2292512110434473e-06
This X-ray shows no signs of disease in the thoracic cavity.: 1.0017811025164747e-09
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 1.1422487261603109e-10
Histopathology slide of Colon adenocarcinoma with glandular structures.: 1.3998058015995095e-14
CT scan showing space-occupying lesion within the brain parenchyma.: 3.9834777898408325e-15
Endoscopic findings consistent with reflux esophagitis.: 1.3123993071020358e-15
Ultrasound image of the breast showing irregular margins and malignant features.: 2.253935085006788e-16
MRI reveals contrast-enhancing brain tumor with mass effect.: 7.685378746366523e-17


pathology.jpeg:
Histopathology slide of Colon adenocarcinoma with glandular structures.: 0.9036254286766052
Ultrasound image of the breast showing irregular margins and malignant features.: 0.06580240279436111
MRI reveals contrast-enhancing brain tumor with mass effect.: 0.013663504272699356
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 0.012562202289700508
CT scan showing space-occupying lesion within the brain parenchyma.: 0.004343675449490547
Endoscopic findings consistent with reflux esophagitis.: 2.7845378554047784e-06
This X-ray shows no signs of disease in the thoracic cavity.: 4.90710228007174e-08
A retinal OCT image of Age-related Macular Degeneration.: 3.1962291147102917e-10
Fundus image showing no signs of diabetic retinopathy.: 1.6641193978372826e-10


endoscopy.jpg:
Endoscopic findings consistent with reflux esophagitis.: 0.9999998807907104
This X-ray shows no signs of disease in the thoracic cavity.: 8.7921250724321e-08
Histopathology slide of Colon adenocarcinoma with glandular structures.: 1.133264770913911e-08
Ultrasound image of the breast showing irregular margins and malignant features.: 4.8003299113474895e-09
MRI reveals contrast-enhancing brain tumor with mass effect.: 3.0636103463127506e-12
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 2.5758850347862294e-12
Fundus image showing no signs of diabetic retinopathy.: 2.3819182303624897e-12
CT scan showing space-occupying lesion within the brain parenchyma.: 2.975548139046852e-13
A retinal OCT image of Age-related Macular Degeneration.: 1.1918031394674705e-13


ct.jpg:
CT scan showing space-occupying lesion within the brain parenchyma.: 0.9880825281143188
MRI reveals contrast-enhancing brain tumor with mass effect.: 0.01191753800958395
A retinal OCT image of Age-related Macular Degeneration.: 1.476712618853071e-08
This X-ray shows no signs of disease in the thoracic cavity.: 1.759521972566347e-09
Fundus image showing no signs of diabetic retinopathy.: 6.355690973514072e-11
Ultrasound image of the breast showing irregular margins and malignant features.: 7.1412862602537874e-12
Histopathology slide of Colon adenocarcinoma with glandular structures.: 2.5253347115815512e-14
Endoscopic findings consistent with reflux esophagitis.: 1.206015335882861e-14
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 6.285185638951881e-17


mri.jpg:
MRI reveals contrast-enhancing brain tumor with mass effect.: 0.9984353184700012
CT scan showing space-occupying lesion within the brain parenchyma.: 0.0015646152896806598
Ultrasound image of the breast showing irregular margins and malignant features.: 5.4712259611733316e-08
This X-ray shows no signs of disease in the thoracic cavity.: 1.4370501233429422e-08
Histopathology slide of Colon adenocarcinoma with glandular structures.: 2.4582424637542566e-11
A retinal OCT image of Age-related Macular Degeneration.: 1.2475623832608473e-11
Fundus image showing no signs of diabetic retinopathy.: 6.942392841158274e-12
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 2.884681154970842e-14
Endoscopic findings consistent with reflux esophagitis.: 2.2693863423932457e-15


ultrasound.png:
Ultrasound image of the breast showing irregular margins and malignant features.: 1.0
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 5.10527797814575e-09
Histopathology slide of Colon adenocarcinoma with glandular structures.: 1.0126093297202488e-09
This X-ray shows no signs of disease in the thoracic cavity.: 5.969303610436905e-12
Endoscopic findings consistent with reflux esophagitis.: 1.4607133861851973e-12
A retinal OCT image of Age-related Macular Degeneration.: 1.247932285937045e-13
CT scan showing space-occupying lesion within the brain parenchyma.: 1.1709451225479228e-13
MRI reveals contrast-enhancing brain tumor with mass effect.: 9.638529530715464e-14
Fundus image showing no signs of diabetic retinopathy.: 3.825065688045333e-15


dermatology.jpeg:
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 1.0
A retinal OCT image of Age-related Macular Degeneration.: 7.261981987971566e-11
Fundus image showing no signs of diabetic retinopathy.: 6.5111973972242776e-12
Ultrasound image of the breast showing irregular margins and malignant features.: 5.396932033219226e-12
Histopathology slide of Colon adenocarcinoma with glandular structures.: 9.821487630882857e-15
Endoscopic findings consistent with reflux esophagitis.: 2.508529577908026e-15
MRI reveals contrast-enhancing brain tumor with mass effect.: 2.463463639935614e-17
CT scan showing space-occupying lesion within the brain parenchyma.: 1.3454762679471163e-17
This X-ray shows no signs of disease in the thoracic cavity.: 1.2205722022467623e-19


oct.jpg:
A retinal OCT image of Age-related Macular Degeneration.: 0.9965401887893677
Fundus image showing no signs of diabetic retinopathy.: 0.0034598938655108213
This X-ray shows no signs of disease in the thoracic cavity.: 9.775484421936653e-09
Benign mole with uniform color and symmetry indicating melanocytic nevus.: 2.7705182592541178e-09
Ultrasound image of the breast showing irregular margins and malignant features.: 1.2160839002461898e-09
Endoscopic findings consistent with reflux esophagitis.: 1.0106535400233874e-11
Histopathology slide of Colon adenocarcinoma with glandular structures.: 4.0847861659983054e-13
CT scan showing space-occupying lesion within the brain parenchyma.: 1.131460324369639e-17
MRI reveals contrast-enhancing brain tumor with mass effect.: 1.5591444647950763e-19
```
</details>



## Citation

If you find our work and this repository helpful, please consider giving our repo a star and citing our paper as follows:

```bibtex
xx
```

## Acknowledgement
Our code repository is mainly built on [UniMedCLIP](https://github.com/mbzuai-oryx/UniMed-CLIP/blob/main/README.md), [MetaCLIP](https://github.com/facebookresearch/MetaCLIP), and [OpenCLIP](https://github.com/mlfoundations/open_clip). We thank the authors for releasing their code. 

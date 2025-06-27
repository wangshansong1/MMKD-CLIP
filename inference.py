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
text_prompts = ["This X-ray shows no signs of disease in the thoracic cavity.",
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
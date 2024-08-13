[DexiNed](https://github.com/xavysp/DexiNed)
The used checkpoint can be downloaed from [this url](https://drive.google.com/file/d/1V56vGTsu7GYiQouCIKvTWl5UKCZ6yCNu/view). The downloaded checkpoint should be putted to `./OUTPUT/DexiNed/checkpoint/10_model.pth`.



## Prepare segmentation model and segmentation mask
1. Segmentation Model

We use [Mask2Former](https://github.com/facebookresearch/Mask2Former) to get the segmentation mask.
The used checkpoint can be downloaed from [this url](https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_swin_base_IN21k_384_bs16_50ep/model_final_54b88a.pkl). The downloaded checkpoint should be putted to `./OUTPUT/Mask2Former/checkpoint/model_final_54b88a.pkl`.

2. Prepare Segmentation Mask For Training
```
# ffhq
python scripts/inference_seg.py --image_dir data/ffhq --save_root data/segmentation/ffhq --gpu 0
python scripts/inference_seg.py --image_dir data/ffhq_256_sample/gt --save_root data/ffhq_256_sample/gt_seg --gpu 0
python scripts/inference_seg.py --image_dir data/ffhq_512_sample/gt --save_root data/ffhq_512_sample/gt_seg --gpu 0


# naturalscene
python scripts/inference_seg.py --image_dir data/naturalscene --save_root data/segmentation/naturalscene --gpu 0
python scripts/inference_seg.py --image_dir data/naturalscene_256_sample/gt --save_root data/naturalscene_256_sample/gt_seg --gpu 0
python scripts/inference_seg.py --image_dir data/naturalscene_512_sample/gt --save_root data/naturalscene_512_sample/gt_seg --gpu 0


# imagenet
python scripts/inference_seg.py --image_dir data/imagenet --save_root data/segmentation/imagenet --gpu 0
python scripts/inference_seg.py --image_dir data/imagenet_256_sample/gt --save_root data/imagenet_256_sample/gt_seg --gpu 0
python scripts/inference_seg.py --image_dir data/imagenet_512_sample/gt --save_root data/imagenet_512_sample/gt_seg --gpu 0

```





## FFHQ
Create a symlink `data/ffhq` pointing to the `images1024x1024` folder obtained
from the [FFHQ repository](https://github.com/NVlabs/ffhq-dataset).

## ImageNet
Download [ILSVRC2012_img_train.tar](https://academictorrents.com/details/a306397ccf9c2ead27155983c254227c0fd938e2) and [ILSVRC2012_img_val.tar](https://academictorrents.com/details/5d6d0df7ed81efd49ca99ea4737e0ae5e3a5f2e5) through [Academic
Torrents](http://academictorrents.com/) if you do not have imagenet dataset. Then put or symlink the ImageNet data into
`$data/imagenet/{split}` where `{split}` is one
of `train`/`validation`. It should have the following structure:
```
$data/imagenet/{split}/
├── n01440764
│   ├── n01440764_10026.JPEG
│   ├── n01440764_10027.JPEG
│   ├── ...
├── n01443537
│   ├── n01443537_10007.JPEG
│   ├── n01443537_10014.JPEG
│   ├── ...
├── ...
```

## Places2_natural
Download [train_split](http://data.csail.mit.edu/places/places365/train_large_places365challenge.tar) and [validation_split](https://www.dropbox.com/s/ttiggl1nrgwutc5/natural_scene_val.zip?dl=1), and orgnize the data as follows:
```
$data/naturalscene/
├── b
│   ├── butte
│   |     |── 00000001.jpg
│   |     |── ...
├── c
│   ├── canyon
│   |     |── 00000001.jpg
│   |     |── ...
├── ...
├── val
│   |── Places365_val00000007.jpg
│   |── ...
```

## irregular mask
We use the mask provided by [PConv](https://nv-adlr.github.io/publication/partialconv-inpainting). Only the testing mask is needed, which can be download from [here](https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip?dl=0). Then orgnize the data as follows:
```
$data/irregular-mask/
├── testing_mask_dataset
│   |── 00000.png
│   |── ...
```


## conceptual caption
Concaptual caption can be downloaded from Azure Storage:

`azcopy copy https://pubdataseteu.blob.core.windows.net/conceptualcaption?sv=2020-04-08&st=2021-06-27T06%3A29%3A59Z&se=2022-12-30T06%3A29%3A00Z&sr=c&sp=rl&sig=%2BwsVmd88L09ePEo2bcWvrE%2FeYnqqaOGIhKD%2FY2wvEjQ%3D ./data --recursive
`

Note that the expiration date of the above link is 2022-12-30.

It should have the following structure:
```
$data/conceptualcaption/{split}/
├── train
│   ├── gcc-train-image-00.tsv
│   ├── gcc-train-image-00.lineidx
│   ├── gcc-train-text-00.tsv
│   ├── gcc-train-ext-00.lineidx
│   ├── gcc-train-image-01.tsv
│   ├── gcc-train-image-01.lineidx
│   ├── gcc-train-text-01.tsv
│   ├── gcc-train-ext-01.lineidx
│   ├── ...
├── val
│   ├── gcc-val-image.tsv
│   ├── gcc-val-image.lineidx
│   ├── gcc-val-text.tsv
│   ├── gcc-val-text.lineidx
│   ├── ...
├── ...
```

## CUB_200_2011
CUB_200_2011 is a bird dataset. Download images from [CUB_200_2011 images and annotations](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html). The text is borrowed from a [paper](https://arxiv.org/abs/1605.05395) of CVPR 2016, which can be download from [here](https://github.com/reedscot/cvpr2016).
Origin caption text can be download from [here](https://github.com/taoxugit/AttnGAN).


## CelebA
- download `CelebA` from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8), only the `Anno` and `Img/img_align_celeba_png.7z` need to be downloaded. 
- unzip `Anno` and `Img/img_align_celeba_png.7z` to `celeba/Anno` and `celeba/img_align_celeba`
- Create a symlink `data/celeba` pointing to the folder `celeba`


## CelebA-HQ
Create a symlink `data/celeba-hq` pointing to a folder containing the `.jpg`
files of CelebA-HQ (instructions to obtain them can be found in the [PGGAN
repository](https://github.com/tkarras/progressive_growing_of_gans)). Before that, you
need to:

- download `CelebA-delta` from [here](https://drive.google.com/drive/folders/0B4qLcYyJmiz0TXY1NG02bzZVRGs)
- download `CelebA` from [here](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8), only the `Anno` and `Img/img_celeba.7z` need to be downloaded. 
- git clone [PGGAN
repository](https://github.com/tkarras/progressive_growing_of_gans) and replace `dataset_tool.py` with our modified `scripts/tools/dataset_tool.py`. There may be some errors while making the environment of PGGAN. Keeping in mind that install pillow with conda and replace tensorflow-gpu with tensorflow since we only need to generate some images rather than trainig a network.

- Finaly, generate images with 
`python dataset_tool.py create_celebahq path/to/save/celeba-hq path/to/celeba path/to/celeba-deltas --num_threads 30`. There should be two folders `imgHQ` and `imgHQ_npy` in `path/to/save/celeba-hq`. Just move or symlink `path/to/save/celeba-hq` to `data/celeba-hq`, Note that `imgHQ_npy` is no need and can be deleted.

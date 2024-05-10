

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
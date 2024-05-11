conda create -n ImgSyn python=3.8
conda activate ImgSyn

# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt

python setup.py build develop --user

########################################################
#            The following is optional 
#  Only required by the controllable image inpainting
########################################################

# choose the right detectron2 according to your cuda and torch versions
# refer to https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html

# compile deformable attention for mask2former
cur_dir=$(pwd)
cd image_synthesis/modeling/modules/mask2former/mask2former/modeling/pixel_decoder/ops
# make sure that the cuda home is avaiable
# export CUDA_HOME=/usr/local/cuda-11.1
sh make.sh
cd $cur_dir


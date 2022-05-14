conda create -n ImgSyn python=3.7
conda activate ImgSyn

conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch

pip install -r requirements.txt

sh install.sh
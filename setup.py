from setuptools import setup, find_packages

setup(
  name = 'image-synthesis',
  packages = find_packages(),
  include_package_data = True,
  version = '0.0.1',
  license='MIT',
  description = 'code base for image synthesis',
  author = 'Qiankun Liu',
  author_email = 'liuqk3@outlook.com',
  url = 'https://github.com/liuqk3/image-synthesis',
  keywords = [
    'artificial intelligence',
    'attention mechanism',
    'transformers',
    'text-to-image',
    'image synthesis'
  ],
  # install_requires=[
  #   # 'axial_positional_embedding',
  #   'einops>=0.3',
  #   'ftfy',
  #   'pillow',
  #   # 'torch>=1.6',
  #   # 'torchvision',
  #   'tqdm'
  # ],
)
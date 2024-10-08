from setuptools import setup, find_packages

setup(
    name='hyperseg',
    version='0.1.0',
    author='Nick Theisen',
    author_email='nicktheisen@uni-koblenz.de',
    packages=find_packages(),
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='A framework for hyperspectral semantic segmentation experiments.',
    long_description=open('README.md').read(),
    install_requires=[
        "einops",
        "h5py",
        "hydra-core",
        "imageio",
        "kornia",
        "matplotlib",
        "opencv-python",
        "pandas",
        "pytorch_lightning",
        "tensorboard",
        "tifffile",
        "torch",
        "torchinfo",
        "torchmetrics>=1.0.0",
        "torchsummary",
        "torchvision",
        "tqdm",
        "scikit-learn",
        "wandb",
        ],
)

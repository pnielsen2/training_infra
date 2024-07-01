from setuptools import setup, find_packages

setup(
    name="training_infra",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'pyro-ppl',
        'wandb',
        'numpy'
    ],
    # Optional
    author="Philip Nielsen",
    author_email="pnielsen2@outlook.com",
    description="ML Training Infrastructure",
    license="MIT",
    keywords="Machine Learning",
)

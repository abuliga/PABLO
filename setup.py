from setuptools import setup, find_packages

setup(
    name='nirdizati_light',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'dice_ml @ git+https://github.com/abuliga/DiCE.git@origin/main',
        'declare4py @ git+https://github.com/abuliga/declare4py.git@main',
        'pymining==0.2',
        'pandas~=1.5.3',
        'pm4py~=2.7.8.2',
        'scikit-learn',
        'hyperopt~=0.2.7',
        'numpy==1.23.2'
        'dateparser~=1.1.8',
        'holidays~=0.28',
        'funcy~=2.0.0',
        'xgboost~=2.0.3',
        'dtreeviz',
        'seaborn',
        'paretoset',
        'category_encoders'
    ]
)

from setuptools import find_packages, setup


setup(name="mini_psp",
      version="1.0.0",
      description="Mini-PSPNet for Urban Land-Use/Land-Cover Classification of Remote Sensing images",
      author="Surya Dheeshjith",
      author_email='Surya.Dheeshjith@gmail.com',
      platforms=["any"],
      url="https://github.com/suryadheeshjith/ISRO_Repo",
      packages=find_packages(exclude=["figures"]),
      install_requires=[
            "numpy==1.18.1",
            "argparse",
            "tensorflow==1.15.2",
            "scikit-learn==0.22.1",
            "scipy==1.4.1",
            "rasterio==1.1.5",
            "matplotlib==3.2.1",
            "json5==0.8.5",
            "tensorflow==1.15.2",
            "logger==1.4"],
        extras_require={
            "testing": ["pytest"]
      }

)

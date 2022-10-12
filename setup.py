from setuptools import setup

setup(
    name="LANDMark",
    version="1.2.0",
    author="Josip Rudar, Teresita M. Porter, Michael Wright, G. Brian Golding, Mehrdad Hajibabaei",
    author_email="rudarj@uoguelph.ca",
    description="LANDMark: An ensemble approach to the supervised selection of biomarkers in high-throughput sequencing data",
    url="https://github.com/jrudar/LANDMark",
    license="MIT",
    keywords="biomarker selection, metagenomics, metabarcoding, biomonitoring, ecological assessment, machine learning, supervised learning, unsupervised learning",
    packages=["LANDMark"],
    python_requires=">=3.10",
    install_requires=[
        "scikit-learn == 1.1.2",
        "joblib == 1.2.0",
        "shap == 0.41.0",
        "numpy == 1.23.3",
        "tensorflow_addons == 0.18.0",
        "tensorflow == 2.10.0",
        "pandas == 1.5.0",
        "scipy ==1.8.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.10+",
        "License :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Ecology :: Biomarker Selection :: Metagenomics :: Supervised Learning :: Unsupervised Learning :: Metabarcoding :: Biomonitoring :: Ecological Assessment :: Machine Learning",
    ],
)

''' 
How to install theano for :
    CUDA12.2
    Python 3.11
    Numpy 1.26.4
    gcc 11.2

0. copy or link my theano directory into
      ..../miniconda3xx/lib/python3.11/site-packages/theano
   also my version of pygpu

1. Install conda, python 3.11
   pytensor, libgpuarray, scikit-cuda, pygpu, etc....

2.  be sure to carry over my changes (marked with PMB) to
   pygpu, skcuda, and  import my libsmegma.so and smegma*.h
   
3. Update system C-compiler:
         yum install devtools-11
         scl enable devtoolset-11 bash
         (run theano in this bash environment)
         Note that the system c-compiler needs to be
         the same or compatible with :

          import sys
          print(sys.version)

4. update scan_perform.c:
          in theano/scan:
          conda install cython
          cython scan_perform.pyx -o c_code/scan_perform.c

5. Here are the packages and versions in the conda environment

6. Be sure to use gcc.cxxflags=-DCUDAVERSION=12 in THEANO_FLAGS:
  export THEANO_FLAGS=mode=FAST_RUN,device=cuda,floatX=float64,magma.library_path=/home/paul.baggenstoss/software/magma-2.5.2/lib,magma.include_path=/home/paul.baggenstoss/software/magma-2.5.2/include,gcc.cxxflags=-I/usr/local/cuda/include,gcc.cxxflags=-DCUDAVERSION=12


7. check package versions using "conda list":
# packages in environment at /home/paul.baggenstoss/miniconda3.11:
#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
_sysroot_linux-64_curr_repodata_hack 3                   haa98f57_10  
anaconda-anon-usage       0.7.0           py311hfc0e8ea_101  
annotated-types           0.6.0           py311h06a4308_0  
archspec                  0.2.3              pyhd3eb1b0_0  
binutils_impl_linux-64    2.40                 h5293946_0  
binutils_linux-64         2.40.0               hc2dff05_2  
blas                      1.0                         mkl  
boltons                   24.1.0          py311h06a4308_0  
brotli-python             1.0.9           py311h6a678d5_9  
bzip2                     1.0.8                h5eee18b_6  
c-ares                    1.19.1               h5eee18b_0  
ca-certificates           2025.4.26            hbd8a1cb_0    conda-forge
certifi                   2025.4.26       py311h06a4308_0  
cffi                      1.17.1          py311h1fdaa30_1  
charset-normalizer        3.3.2              pyhd3eb1b0_0  
conda                     25.3.1          py311h38be061_1    conda-forge
conda-anaconda-telemetry  0.1.2           py311h06a4308_1  
conda-anaconda-tos        0.1.3           py311h06a4308_0  
conda-content-trust       0.2.0           py311h06a4308_1  
conda-libmamba-solver     25.4.0             pyhd3eb1b0_0  
conda-package-handling    2.4.0           py311h06a4308_0  
conda-package-streaming   0.11.0          py311h06a4308_0  
cons                      0.4.6           py311h06a4308_0  
contourpy                 1.3.1           py311hdb19cb5_0  
cpp-expected              1.1.0                hdb19cb5_0  
cryptography              44.0.1          py311h7825ff9_0  
cuda-cccl                 12.4.127                      0    nvidia
cuda-compiler             12.4.1                        0    nvidia
cuda-cudart               12.4.127                      0    nvidia
cuda-cudart-dev           12.4.127                      0    nvidia
cuda-cuobjdump            12.4.127                      0    nvidia
cuda-cuxxfilt             12.4.127                      0    nvidia
cuda-libraries            12.4.1                        0    nvidia
cuda-nvcc                 12.4.131                      0    nvidia
cuda-nvprune              12.4.127                      0    nvidia
cuda-nvrtc                12.4.127                      0    nvidia
cuda-opencl               12.4.127                      0    nvidia
cuda-version              12.1                 h1f5ad73_3  
cudnn                     8.9.2.26               cuda12_0  
cycler                    0.11.0             pyhd3eb1b0_0  
cyrus-sasl                2.1.28               h52b45da_1  
cython                    3.1.1                    pypi_0    pypi
distro                    1.9.0           py311h06a4308_0  
etuples                   0.3.9           py311h06a4308_0  
expat                     2.7.1                h6a678d5_0  
filelock                  3.17.0          py311h06a4308_0  
fmt                       9.1.0                hdb19cb5_1  
fontconfig                2.14.1               h55d465d_3  
fonttools                 4.55.3          py311h5eee18b_0  
freetype                  2.13.3               h4a9f257_0  
frozendict                2.4.2           py311h06a4308_0  
fsspec                    2025.5.0                 pypi_0    pypi
gcc_impl_linux-64         11.2.0               h1234567_1  
gcc_linux-64              11.2.0               h5c386dc_2  
gxx_impl_linux-64         11.2.0               h1234567_1  
gxx_linux-64              11.2.0               hc2dff05_2  
icu                       73.1                 h6a678d5_0  
idna                      3.7             py311h06a4308_0  
intel-openmp              2023.1.0         hdb19cb5_46306  
jinja2                    3.1.6                    pypi_0    pypi
joblib                    1.4.2           py311h06a4308_0  
jpeg                      9e                   h5eee18b_3  
jsonpatch                 1.33            py311h06a4308_1  
jsonpointer               2.1                pyhd3eb1b0_0  
kernel-headers_linux-64   3.10.0              h57e8cba_10  
kiwisolver                1.4.8           py311h6a678d5_0  
krb5                      1.20.1               h143b758_1  
lcms2                     2.16                 h92b89f2_1  
ld_impl_linux-64          2.40                 h12ee557_0  
lerc                      4.0.0                h6a678d5_0  
libabseil                 20250127.0      cxx17_h6a678d5_0  
libarchive                3.7.7                hfab0078_0  
libcublas                 12.4.5.8                      0    nvidia
libcufft                  11.2.1.3                      0    nvidia
libcufile                 1.9.1.3                       0    nvidia
libcups                   2.4.2                h2d74bed_1  
libcurand                 10.3.5.147                    0    nvidia
libcurl                   8.12.1               hc9e6f67_0  
libcusolver               11.6.1.9                      0    nvidia
libcusparse               12.3.1.170                    0    nvidia
libcusparse-dev           12.2.0.103                    0    nvidia
libdeflate                1.22                 h5eee18b_0  
libedit                   3.1.20230828         h5eee18b_0  
libev                     4.33                 h7f8727e_1  
libffi                    3.4.4                h6a678d5_1  
libgcc-devel_linux-64     11.2.0               h1234567_1  
libgcc-ng                 11.2.0               h1234567_1  
libgfortran-ng            11.2.0               h00389a5_1  
libgfortran5              11.2.0               h1234567_1  
libglib                   2.78.4               hdc74915_0  
libgomp                   11.2.0               h1234567_1  
libgpuarray               0.7.6                h7f8727e_1  
libiconv                  1.16                 h5eee18b_3  
libmamba                  2.0.5                haf1ee3a_1  
libmambapy                2.0.5           py311hdb19cb5_1  
libnghttp2                1.57.0               h2d74bed_0  
libnpp                    12.2.5.30                     0    nvidia
libnvfatbin               12.4.127                      0    nvidia
libnvjitlink              12.9.41                       0    nvidia
libnvjpeg                 12.3.1.117                    0    nvidia
libpng                    1.6.39               h5eee18b_0  
libpq                     17.4                 hdbd6064_0  
libprotobuf               5.29.3               hc99497a_0  
libsolv                   0.7.30               he621ea3_1  
libssh2                   1.11.1               h251f7ec_0  
libstdcxx-devel_linux-64  11.2.0               h1234567_1  
libstdcxx-ng              13.2.0               hc0a3c3a_7    conda-forge
libtiff                   4.7.0                hde9077f_0  
libuuid                   1.41.5               h5eee18b_0  
libwebp-base              1.3.2                h5eee18b_1  
libxcb                    1.17.0               h9b100fa_0  
libxkbcommon              1.9.1                h69220b7_0  
libxml2                   2.13.8               hfdd30dd_0  
logical-unification       0.4.6           py311h06a4308_0  
lz4-c                     1.9.4                h6a678d5_1  
mako                      1.3.10                   pypi_0    pypi
markdown-it-py            2.2.0           py311h06a4308_1  
markupsafe                3.0.2           py311h5eee18b_0  
matplotlib                3.10.0          py311h06a4308_0  
matplotlib-base           3.10.0          py311hbfdbfaf_0  
mdurl                     0.1.0           py311h06a4308_0  
menuinst                  2.2.0           py311h06a4308_1  
minikanren                1.0.3           py311h06a4308_0  
mkl                       2023.1.0         h213fc3f_46344  
mkl-service               2.4.0           py311h5eee18b_2  
mkl_fft                   1.3.11          py311h5eee18b_0  
mkl_random                1.2.8           py311ha02d727_0  
mpmath                    1.3.0                    pypi_0    pypi
multipledispatch          0.6.0           py311h06a4308_0  
mysql                     8.4.0                h721767e_2  
ncurses                   6.4                  h6a678d5_0  
networkx                  3.4.2                    pypi_0    pypi
nlohmann_json             3.11.2               h6a678d5_0  
numpy                     1.26.4          py311h08b1b3b_0  
numpy-base                1.26.4          py311hf175353_0  
nvidia-cublas-cu12        12.4.5.8                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.4.127                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.4.127                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.4.127                 pypi_0    pypi
nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
nvidia-cufft-cu12         11.2.1.3                 pypi_0    pypi
nvidia-curand-cu12        10.3.5.147               pypi_0    pypi
nvidia-cusolver-cu12      11.6.1.9                 pypi_0    pypi
nvidia-cusparse-cu12      12.3.1.170               pypi_0    pypi
nvidia-cusparselt-cu12    0.6.2                    pypi_0    pypi
nvidia-nccl-cu12          2.21.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.4.127                 pypi_0    pypi
nvidia-nvtx-cu12          12.4.127                 pypi_0    pypi
openjpeg                  2.5.2                h0d4d230_1  
openldap                  2.6.4                h42fbc30_0  
openssl                   3.0.16               h5eee18b_0  
packaging                 24.2            py311h06a4308_0  
pcre2                     10.42                hebb0a14_1  
pillow                    11.1.0          py311hac6e08b_1  
pip                       25.0            py311h06a4308_0  
platformdirs              4.3.7           py311h06a4308_0  
pluggy                    1.5.0           py311h06a4308_0  
pthread-stubs             0.3                  h0ce48e5_1  
pybind11-abi              4                    hd3eb1b0_1  
pycosat                   0.6.6           py311h5eee18b_2  
pycparser                 2.21               pyhd3eb1b0_0  
pycuda                    2025.1                   pypi_0    pypi
pydantic                  2.10.3          py311h06a4308_0  
pydantic-core             2.27.1          py311h4aa5aa6_0  
pygments                  2.19.1          py311h06a4308_0  
pygpu                     0.7.6           py311hbed6279_1  
pyparsing                 3.2.0           py311h06a4308_0  
pyqt                      6.7.1           py311h6a678d5_1  
pyqt6-sip                 13.9.1          py311h5eee18b_1  
pysocks                   1.7.1           py311h06a4308_0  
pytensor                  2.23.0          py311ha02d727_1  
python                    3.11.11              he870216_0  
python-dateutil           2.9.0post0      py311h06a4308_2  
python_abi                3.11                    2_cp311    conda-forge
pytools                   2025.1.5                 pypi_0    pypi
qtbase                    6.7.3                hdaa5aa8_0  
qtdeclarative             6.7.3                h6a678d5_0  
qtsvg                     6.7.3                he621ea3_0  
qttools                   6.7.3                h80c7b02_0  
qtwebchannel              6.7.3                h6a678d5_0  
qtwebsockets              6.7.3                h6a678d5_0  
readline                  8.2                  h5eee18b_0  
regex                     2024.11.6                pypi_0    pypi
reproc                    14.2.4               h6a678d5_2  
reproc-cpp                14.2.4               h6a678d5_2  
requests                  2.32.3          py311h06a4308_1  
rich                      13.9.4          py311h06a4308_0  
ruamel.yaml               0.18.10         py311h5eee18b_0  
ruamel.yaml.clib          0.2.12          py311h5eee18b_0  
scikit-cuda               0.5.3                    pypi_0    pypi
scikit-learn              1.6.1           py311h6a678d5_0  
scipy                     1.15.3          py311h525edd1_0  
setuptools                78.1.1          py311h06a4308_0  
simdjson                  3.10.1               hdb19cb5_0  
sip                       6.10.0          py311h6a678d5_0  
siphash24                 1.7                      pypi_0    pypi
six                       1.17.0          py311h06a4308_0  
spdlog                    1.11.0               hdb19cb5_0  
sqlite                    3.45.3               h5eee18b_0  
sympy                     1.13.1                   pypi_0    pypi
sysroot_linux-64          2.17                h57e8cba_10  
tbb                       2021.8.0             hdb19cb5_0  
threadpoolctl             3.5.0           py311h92b7b1e_0  
tiktoken                  0.9.0                    pypi_0    pypi
tk                        8.6.14               h39e8969_0  
toolz                     1.0.0           py311h06a4308_0  
torch                     2.6.0                    pypi_0    pypi
tornado                   6.4.2           py311h5eee18b_0  
tqdm                      4.67.1          py311h92b7b1e_0  
triton                    3.2.0                    pypi_0    pypi
truststore                0.10.0          py311h06a4308_0  
typing-extensions         4.12.2          py311h06a4308_0  
typing_extensions         4.12.2          py311h06a4308_0  
tzdata                    2025a                h04d1e81_0  
unicodedata2              15.1.0          py311h5eee18b_1  
urllib3                   2.3.0           py311h06a4308_0  
wheel                     0.45.1          py311h06a4308_0  
xcb-util                  0.4.1                h5eee18b_2  
xcb-util-cursor           0.1.5                h5eee18b_0  
xcb-util-image            0.4.0                h5eee18b_2  
xcb-util-renderutil       0.3.10               h5eee18b_0  
xkeyboard-config          2.44                 h5eee18b_0  
xorg-libx11               1.8.12               h9b100fa_1  
xorg-libxau               1.0.12               h9b100fa_0  
xorg-libxdmcp             1.1.5                h9b100fa_0  
xorg-xorgproto            2024.1               h5eee18b_1  
xz                        5.6.4                h5eee18b_1  
yaml-cpp                  0.8.0                h6a678d5_1  
zlib                      1.2.13               h5eee18b_1  
zstandard                 0.23.0          py311h2c38b39_1  
zstd                      1.5.6                hc292b87_0  
'''

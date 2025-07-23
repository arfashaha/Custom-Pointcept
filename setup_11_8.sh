# 1. Create environment and activate
conda create -n geom3d python=3.8 -y
conda activate geom3d

# 2. Set CUDA environment variables (add to ~/.bashrc if you want persistence)
export CUDA_HOME=$HOME/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 3. Basic scientific & PyTorch deps
conda install openblas-devel ninja h5py pyyaml -c anaconda -y
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# need to install libffi 3.2.1 because openblas devel make libfii in newest versions 
# and will cause error when importing torch
conda install libffi=3.2.1 -y

# then need to downgrade gcc to 11 
conda install gcc=11 gxx=11 -c conda-forge -y
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
$CC --version

this will cause issue because cluster gcc version is 13, when installing 11 we will be broken to python graal

# 4. (Optional, but often needed) General Python ML deps
pip install sharedarray tensorboard tensorboardx wandb yapf addict einops scipy plyfile termcolor timm

# 5. Torch geometric core deps
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric

# 6. MinkowskiEngine (from source if needed)
# If you have the repo:
# git clone https://github.com/NVIDIA/MinkowskiEngine.git
# cd MinkowskiEngine
# python setup.py install --force_cuda --blas=openblas --blas_include_dirs=${CONDA_PREFIX}/include
# cd ..
# Or just pip (sometimes works fine):
pip install -U git+https://github.com/NVIDIA/MinkowskiEngine

# 7. Torch-Points3D
git clone https://github.com/nicolas-chaulet/torch-points3d.git
cd torch-points3d
pip install hydra-core==1.1 omegaconf==2.1.1 pyyaml
pip install -r requirements_clean.txt

# 8. (If you installed extra torch/geometric versions above, reinstall right ones)
pip uninstall -y torch torchvision torchaudio torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# 9. Fix Numba incompatibility (for some point cloud tools)
conda install -c conda-forge numba=0.56.4 llvmlite=0.39.1 -y

# 10. spconv (SparseUNet, optional, only if needed for your projects)
pip install spconv-cu118

# 11. PPT (CLIP)
pip install ftfy regex tqdm
pip install git+https://github.com/openai/CLIP.git

# 12. PointOps (PTv1, PTv2, precise eval) -- if you have this repo
# cd libs/pointops
# export TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"   # Edit as needed for your GPU
# python setup.py install
# cd ../..

# 13. Open3D (visualization, optional)
pip install open3d

# 14. (Optional) Confirm CUDA version
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

echo "✅ 3D Deep Learning environment (geom3d) ready!"

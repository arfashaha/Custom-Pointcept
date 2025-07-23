# 1. Create environment
conda create -n geom3d python=3.8 -y
conda activate geom3d

export CUDA_HOME=$HOME/cuda-11.3
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
$CUDA_HOME/bin/nvcc --version
chmod +x cuda_11.3.0_520.61.05_linux.run
mkdir -p $HOME/cuda-11.8
./cuda_11.3.0_520.61.05_linux.run --silent --toolkit --toolkitpath=$HOME/cuda-11.3
echo 'export CUDA_HOME=$HOME/cuda-11.3' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
$CUDA_HOME/bin/nvcc --version

# 2. PyTorch 1.10.0 + CUDA 11.3
conda install pytorch=1.10.0 torchvision=0.11.1 torchaudio=0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# 3. PyG deps (all from the *special* wheel)
pip install torch-scatter==2.0.9 torch-sparse==0.6.12 torch-cluster==1.5.9 torch-spline-conv==1.2.1 torch-geometric==1.7.2 -f https://data.pyg.org/whl/torch-1.10.0+cu113.html

# 4. torch-points3d and friends
pip install torch-points3d==1.3.0 torch-points-kernels==0.6.10

# 5. MinkowskiEngine (if needed)
pip install -U MinkowskiEngine==0.5.4 --find-links https://nvidia.github.io/MinkowskiEngine/whl/torch-1.10/

# 6. Open3D (optional)
pip install open3d==0.12.0

# 7. Miscellaneous
pip install addict tensorboard tensorboardX wandb einops scipy plyfile termcolor timm yapf

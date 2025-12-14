####### for MacOS
brew install git git-lfs
git lfs install
brew install ninja
ninja --version

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate safetensors huggingface_hub
cd models
# The model is ~15GB model, 20GB+ free disk space required
git lfs clone https://huggingface.co/openvla/openvla-7b 


####### for Ubuntu
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
ninja --version; echo $?  # validate Ninja library for loading model
git lfs clone https://huggingface.co/openvla/openvla-7b
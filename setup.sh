# Thanks to Bryan Li for the structure of the file:  https://github.com/bryanlimy/mri-super-resolution
#!/bin/sh

macOS=false
current_dir="$(pwd -P)"

check_requirements() {
  case $(uname -s) in
    Darwin)
      printf "Installing on macOS"
      export CFLAGS='-stdlib=libc++'
      macOS=true
      ;;
    Linux)
      printf "Installing on Linux"
      ;;
    *)
      printf "Only Linux and macOS are currently supported.\n"
      exit 1
      ;;
  esac
}

install_python_packages() {
  printf "\nInstalling Packages...\n"
  conda install -q poetry -y
  conda install -q cython -y
  conda install -q tqdm -y
  # This whole process won't be necessary once aposteriori is published on pip
  git clone https://github.com/wells-wood-research/aposteriori.git
  cd aposteriori
  poetry install
  cd ..
  if [ $macOS = "false" ]; then
    conda install cudatoolkit cudnn cupti
  fi
  conda install -q tensorflow -y
  conda install -q  h5py -y
}

set_python_path() {
  printf "\nSet conda environment variables...\n"
  conda env config vars set PYTHONPATH=$PYTHONPATH:$current_dir
  export PYTHONPATH=.
}

check_requirements
install_python_packages
set_python_path

printf '\nSetup completed. Please restart your conda environment\n'
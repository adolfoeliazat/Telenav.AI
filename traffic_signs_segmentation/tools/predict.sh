#Cuda visible devices
CUDA_VISIBLE_DEVICES=0

# Input images folder
INPUT_PATH=/data/
# Output results path
OUTPUT_PATH=/data/output/

# Do not modify
CONFIG=./config
UTILS=./utils
ML_FOLDER=./../

cd $ML_FOLDER

PYTHONPATH="${PYTHONPATH}:../:../apollo_python_common/protobuf/:./utils/:./"
export PYTHONPATH

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python3 -u ./inference_server.py --input_path $INPUT_PATH --output_path $OUTPUT_PATH

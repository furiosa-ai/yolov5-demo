# Setup

## Setup environment
```bash
git clone git@github.com:furiosa-ai/yolov5-demo.git
cd yolov5-demo
conda activate <env>
pip install -r requirements.txt
./build.sh
```

## Files

Download weights from [here](https://drive.google.com/file/d/1Cdvld9ASNpnMUAVC10aDSNUBeLYlFBhB/view?usp=sharing) and extract to "./weights"

## Run

```bash
# Arguments
python demo.py -h
  --input INPUT         Path to video file
  --model MODEL         Path to onnx weights
  --no_display          Disables displaying results on the screen (useful for server)
  --frame_limit FRAME_LIMIT
                        Stops inference after frame limit is reached
  --record_file RECORD_FILE
                        Record results to specified video file
```

Example
```bash
# Run yolov5 object detector
python demo.py \
  --input data/test_img.jpg \
  --model weights/yolov5m_warboy_bdd100k_640
```

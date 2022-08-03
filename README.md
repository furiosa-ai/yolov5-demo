# Setup

## Setup environment
```bash
sudo apt-get install ffmpeg 
git clone git@github.com:furiosa-ai/yolov5-demo.git
cd yolov5-demo
conda activate <env>
pip install -r requirements.txt
./build.sh
```

## Weights

Download weights from [here](https://drive.google.com/file/d/1Cdvld9ASNpnMUAVC10aDSNUBeLYlFBhB/view?usp=sharing) and extract to "./weights"

## Calibration
Download BDD100K dataset from [here](https://doc.bdd100k.com/download.html) or prepare own dataset for calibration

## Run

```bash
# Arguments
python demo.py -h
  --input INPUT         Path to image / image folder / video file
  --model MODEL         Path to onnx weights folder
  --framework {onnx,furiosa}
                        Which backend to use
  --calib-data CALIB_DATA
                        Path to calibration data containing image files
  --calib-data-count CALIB_DATA_COUNT
                        How many images to use for calibration
  --no-display          Disables displaying results on the screen (useful for server)
  --frame-limit FRAME_LIMIT
                        Stops inference after frame limit is reached
  --record-file RECORD_FILE
                        Record results to specified video file (e.g. "out.mp4")
```

Example
```bash
# Run yolov5 object detector using OnnxRuntime (f32)
python demo.py \
  --input data/test_img.jpg \
  --model weights/yolov5m_warboy_bdd100k_640 \
  --framework onnx \
  --no-display

# Run yolov5 object detector using FursioaSDK (i8)
python demo.py \
  --input data/test_img.jpg \
  --model weights/yolov5m_warboy_bdd100k_640 \
  --framework furiosa \
  --no-display \
  --calib-data ../data/bdd100k/images/100k/test \
  --calib-data-count 10
```

cd utils/box_decode/cbox_decode
rm -rf build cbox_decode.so
python setup.py build_ext --inplace
cd -
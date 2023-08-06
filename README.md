# 实现自己的tf算子

## 自定义tf算子
* example: b = 2 * a

## 构建
```bash
mkdir build
cd build
cmake ..
make
cp ./libtf_custom_op.so ../
```

## 测试
```bash
# 测试example并导出为onnx
python3 kernel_example_test.py
```
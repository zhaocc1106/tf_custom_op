import tensorflow as tf
import tf2onnx
from onnx import helper

_TENSORFLOW_DOMAIN = "ai.onnx.converters.tensorflow"

twice_op_module = tf.load_op_library('./libtf_custom_op.so')


@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)])
def twice_op(x):
    return twice_op_module.example(x)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()

    def build(self, input_shape):
        print('build with input_shape: {}'.format(input_shape))
        super(MyModel, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        return twice_op(inputs)


if __name__ == '__main__':
    model = MyModel()
    with tf.device('cpu'):
        a = tf.ones([3, 3])
        b = model(a)
        print('calc with cpu op, a: {}\nb: {}'.format(a, b))
    with tf.device('gpu:0'):
        a = tf.ones([3, 3])
        b = model(a)
        print('calc with gpu op, a: {}\nb: {}'.format(a, b))

    # Save to onnx
    model_onnx_path = 'model.onnx'
    print('Exporting model to ONNX format...')
    tf2onnx.convert.from_keras(model, input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.float32)],
                               opset=12, output_path=model_onnx_path, custom_ops={'Example': _TENSORFLOW_DOMAIN})
    print('Model exported to ' + model_onnx_path)

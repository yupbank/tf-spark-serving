import tensorflow as tf

import pyspark.sql.types as T

TF_TO_SPARK_TYPE_MAPPINGS = {
    tf.float32: T.FloatType,
    tf.float64: T.DoubleType,
    tf.int32: T.IntegerType,
    tf.int64: T.LongType,
    tf.string: T.StringType
}


def tf_shape_as_array(shape, elem_type):
    if not shape:
        return elem_type()
    else:
        return T.ArrayType(_tf_shape_as_array(shape[1:], elem_type))


def get_model_full_path(model_base_path, version=None):
    versions = {}
    model_base_path = model_base_path.rstrip('/')
    for version in tf.gfile.ListDirectory(model_base_path):
        version = version.rstrip('/')
        if version.isdigit():
            versions[int(version)] = '/'.join([model_base_path, version])
    if version is None:
        version = max(versions.keys())
    return versions[int(version)]

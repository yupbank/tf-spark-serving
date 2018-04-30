from tss import tf_serving_with_dataframe, export_saved_model


# save a factorial model with output_dir='gs://factorial_model', model_version=1
def save():
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.double, shape=[None])
        result = tf.exp(tf.lgamma(x + 1))
        export_saved_model({'x': x}, {'result': result}, output_dir, model_version, g)

# serve a factorial model

def serve():
    return tf_serving_with_dataframe(df, 'gs://factorial_model')

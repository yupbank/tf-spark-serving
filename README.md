# Tensorflow Serving with spark dataframe Made easy

---

Tensorflow/serving is good at real time serving. Somehow also support batching inferencing, which requires user push batch data to the service.

Meanwhile it might be easier with just bring code to the data.


```python
from tss import tf_serving_with_dataframe

def main():
    pred_df = tf_serving_with_dataframe(df, 'model-base-path')

```


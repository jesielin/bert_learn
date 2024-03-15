from datasets import list_metrics,load_metric

metrics = list_metrics()
# print(metrics)
# print(len(metrics))

metric = load_metric("cola")
print(metric)
# print(metric.inputs_description)
import pyarrow as pa

with open('../../data/mmlu/dev/data-00000-of-00001.arrow', 'rb') as f:
    reader = pa.ipc.RecordBatchStreamReader(f)
    for batch in reader:
        tmp_dict = batch.to_pydict()
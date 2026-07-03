import pyarrow.parquet as pq
import pandas as pd

# Use pyarrow metadata for size info without loading into memory
for fname in ['train_part_1.parquet', 'train_part_2.parquet', 'train_part_3.parquet',
              'pretrain_part_1.parquet', 'pretrain_part_2.parquet', 'pretrain_part_3.parquet',
              'pretest.parquet', 'test.parquet']:
    pf = pq.ParquetFile(fname)
    print(f'{fname}: rows={pf.metadata.num_rows}, cols={pf.metadata.num_columns}')

print()
# Schema of train
pf = pq.ParquetFile('train_part_1.parquet')
print('=== TRAIN SCHEMA ===')
print(pf.schema)

# Read just first 5 rows
print()
print('=== TRAIN PART 1 HEAD ===')
df = pf.read_row_group(0).to_pandas().head(5)
print(df)
print()
print(df.dtypes)

# Labels
print()
tl = pd.read_parquet('train_labels.parquet')
print(f'Labels: {tl.shape}')
print(f'Target: 1={sum(tl.target==1)}, 0={sum(tl.target==0)}')
print(f'Unique customers with labels: {tl.customer_id.nunique()}')

# Test
print()
test = pd.read_parquet('test.parquet')
print(f'Test: {test.shape}')
print(f'Unique test customers: {test.customer_id.nunique()}')
print(f'Test dates: {test.event_dttm.min()} to {test.event_dttm.max()}')

# Pretest
print()
pt = pd.read_parquet('pretest.parquet')
print(f'Pretest: {pt.shape}')
print(f'Pretest dates: {pt.event_dttm.min()} to {pt.event_dttm.max()}')
print(f'Unique pretest customers: {pt.customer_id.nunique()}')

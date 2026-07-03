import pandas as pd
import numpy as np

# Look at train data structure
print('=== TRAIN PART 1 ===')
t1 = pd.read_parquet('train_part_1.parquet')
print(t1.shape)
print(t1.columns.tolist())
print(t1.dtypes)
print(t1.head(3))
print()

print(f'Unique customers in train_part_1: {t1.customer_id.nunique()}')
print()

# Look at pretest  
print('=== PRETEST ===')
pt = pd.read_parquet('pretest.parquet')
print(pt.shape)
print(f'Unique customers in pretest: {pt.customer_id.nunique()}')
print()

# Check test
test = pd.read_parquet('test.parquet')
print(f'Unique customers in test: {test.customer_id.nunique()}')
print(f'Total test events: {len(test)}')
print()

# Check overlap between train labels and train data
tl = pd.read_parquet('train_labels.parquet')
print(f'Unique customers in labels: {tl.customer_id.nunique()}')
print(f'Unique events in labels: {tl.event_id.nunique()}')
print()

# Check if label event_ids are in train part 1
overlap = tl[tl.event_id.isin(t1.event_id)]
print(f'Label events found in train_part_1: {len(overlap)}')

# Look at date distribution in train
print()
print('=== DATE RANGES ===')
print(f"Train part 1 dates: {t1.event_dttm.min()} to {t1.event_dttm.max()}")
print(f"Pretest dates: {pt.event_dttm.min()} to {pt.event_dttm.max()}")
print(f"Test dates: {test.event_dttm.min()} to {test.event_dttm.max()}")

# Memory cleanup
del t1
print()

# Total train events across all parts
t2 = pd.read_parquet('train_part_2.parquet')
t3 = pd.read_parquet('train_part_3.parquet')
print(f'Train part 2: {t2.shape}')
print(f'Train part 3: {t3.shape}')
print(f'Unique customers t2: {t2.customer_id.nunique()}, t3: {t3.customer_id.nunique()}')
del t2, t3

# Pretrain sizes
for i in range(1, 4):
    p = pd.read_parquet(f'pretrain_part_{i}.parquet')
    print(f'Pretrain part {i}: {p.shape}, customers: {p.customer_id.nunique()}')
    del p

"""
Этап 1: Feature Engineering
Обрабатываем данные по частям (чанками) чтобы влезть в 30GB RAM
"""

import numpy as np
import pandas as pd
import gc
from collections import defaultdict


def process_chunk_features(df, prefix=''):
    """Агрегатные фичи по клиенту из одного чанка данных."""
    
    # Группировка по клиенту
    agg_dict = {}
    
    # --- Числовые фичи ---
    # Сумма операций
    amt_agg = df.groupby('customer_id')['operaton_amt'].agg(
        ['count', 'sum', 'mean', 'std', 'median', 'min', 'max']
    )
    amt_agg.columns = [f'{prefix}amt_{c}' for c in amt_agg.columns]
    agg_dict['amt'] = amt_agg
    
    # --- Категориальные фичи ---
    # Кол-во уникальных значений
    for col in ['event_type_nm', 'event_desc', 'channel_indicator_type', 
                'channel_indicator_sub_type', 'mcc_code', 'currency_iso_cd']:
        if col in df.columns:
            nunique = df.groupby('customer_id')[col].nunique()
            agg_dict[f'{col}_nunique'] = nunique.rename(f'{prefix}{col}_nunique')
    
    # --- Безопасность ---
    # VoIP, RDP, compromised, developer_tools
    for col in ['phone_voip_call_state', 'web_rdp_connection']:
        if col in df.columns:
            has_flag = df.groupby('customer_id')[col].agg(
                lambda x: (x == 1).sum() if x.dtype in ['float64', 'int64'] else 0
            )
            agg_dict[f'{col}_count'] = has_flag.rename(f'{prefix}{col}_cnt')
    
    for col in ['compromised', 'developer_tools']:
        if col in df.columns:
            has_flag = df.groupby('customer_id')[col].agg(
                lambda x: (x.astype(str) == '1').sum()
            )
            agg_dict[f'{col}_count'] = has_flag.rename(f'{prefix}{col}_cnt')
    
    # --- Временные фичи ---
    if 'event_dttm' in df.columns:
        df['event_dttm_parsed'] = pd.to_datetime(df['event_dttm'], errors='coerce')
        
        # Час и день недели
        df['hour'] = df['event_dttm_parsed'].dt.hour
        df['dayofweek'] = df['event_dttm_parsed'].dt.dayofweek
        
        # Операции в ночное время (0-6 утра)
        df['is_night'] = (df['hour'] < 6).astype(int)
        night_ratio = df.groupby('customer_id')['is_night'].mean()
        agg_dict['night'] = night_ratio.rename(f'{prefix}night_ratio')
        
        # Операции в выходные
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        weekend_ratio = df.groupby('customer_id')['is_weekend'].mean()
        agg_dict['weekend'] = weekend_ratio.rename(f'{prefix}weekend_ratio')
        
        # Среднее время между операциями (в часах)
        time_diffs = df.sort_values('event_dttm_parsed').groupby('customer_id')['event_dttm_parsed'].agg(
            lambda x: x.diff().dt.total_seconds().mean() / 3600 if len(x) > 1 else np.nan
        )
        agg_dict['time_diff'] = time_diffs.rename(f'{prefix}avg_hours_between_ops')
        
        # Распределение по часам (энтропия)
        hour_entropy = df.groupby('customer_id')['hour'].agg(
            lambda x: -(np.log2(x.value_counts(normalize=True) + 1e-10) * x.value_counts(normalize=True)).sum()
        )
        agg_dict['hour_entropy'] = hour_entropy.rename(f'{prefix}hour_entropy')
        
        # Cleanup
        df.drop(['event_dttm_parsed', 'hour', 'dayofweek', 'is_night', 'is_weekend'], 
                axis=1, inplace=True, errors='ignore')
    
    # --- Канал ---
    if 'channel_indicator_type' in df.columns:
        # Самый частый канал
        mode_channel = df.groupby('customer_id')['channel_indicator_type'].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1
        )
        agg_dict['mode_channel'] = mode_channel.rename(f'{prefix}mode_channel')
    
    # --- POS code ---
    if 'pos_cd' in df.columns:
        pos_nunique = df.groupby('customer_id')['pos_cd'].nunique()
        agg_dict['pos_nunique'] = pos_nunique.rename(f'{prefix}pos_cd_nunique')
    
    # --- Session ---
    if 'session_id' in df.columns:
        session_nunique = df.groupby('customer_id')['session_id'].nunique()
        agg_dict['session_nunique'] = session_nunique.rename(f'{prefix}session_nunique')
    
    # --- Timezone ---
    if 'timezone' in df.columns:
        tz_nunique = df.groupby('customer_id')['timezone'].nunique()
        agg_dict['tz_nunique'] = tz_nunique.rename(f'{prefix}tz_nunique')
    
    # --- OS ---
    if 'operating_system_type' in df.columns:
        os_nunique = df.groupby('customer_id')['operating_system_type'].nunique()
        agg_dict['os_nunique'] = os_nunique.rename(f'{prefix}os_nunique')
    
    # Собираем всё в один датафрейм
    result = pd.concat(list(agg_dict.values()), axis=1)
    return result


def build_event_level_features(df):
    """Фичи на уровне конкретной операции (для train/test)."""
    features = pd.DataFrame(index=df.index)
    
    features['operaton_amt'] = df['operaton_amt']
    features['event_type_nm'] = df['event_type_nm']
    features['event_desc'] = df['event_desc']
    features['channel_indicator_type'] = df['channel_indicator_type']
    features['channel_indicator_sub_type'] = df['channel_indicator_sub_type']
    
    # Parse datetime
    dt = pd.to_datetime(df['event_dttm'], errors='coerce')
    features['hour'] = dt.dt.hour
    features['dayofweek'] = dt.dt.dayofweek
    features['day'] = dt.dt.day
    features['month'] = dt.dt.month
    features['is_night'] = (features['hour'] < 6).astype(int)
    features['is_weekend'] = (features['dayofweek'] >= 5).astype(int)
    
    # Числовые
    features['currency_iso_cd'] = df['currency_iso_cd']
    features['pos_cd'] = df['pos_cd']
    features['timezone'] = df['timezone']
    features['operating_system_type'] = df['operating_system_type']
    
    # Безопасность
    features['phone_voip_call_state'] = df['phone_voip_call_state']
    features['web_rdp_connection'] = df['web_rdp_connection']
    features['compromised'] = pd.to_numeric(df['compromised'], errors='coerce')
    features['developer_tools'] = pd.to_numeric(df['developer_tools'], errors='coerce')
    
    # MCC (закодируем как число)
    features['mcc_code'] = pd.Categorical(df['mcc_code']).codes
    
    # Screen size - разбить на ширину и высоту
    if 'screen_size' in df.columns:
        screen = df['screen_size'].str.extract(r'(\d+)[xX×](\d+)')
        features['screen_w'] = pd.to_numeric(screen[0], errors='coerce')
        features['screen_h'] = pd.to_numeric(screen[1], errors='coerce')
    
    # Battery
    if 'battery' in df.columns:
        features['battery'] = pd.to_numeric(df['battery'], errors='coerce')
    
    # Language features  
    if 'accept_language' in df.columns:
        features['accept_lang_code'] = pd.Categorical(df['accept_language']).codes
    if 'browser_language' in df.columns:
        features['browser_lang_code'] = pd.Categorical(df['browser_language']).codes
    
    return features


def build_relative_features(event_features, customer_agg, customer_ids):
    """
    Relative features: отклонение текущей операции от профиля клиента.
    Ключевые для антифрода!
    """
    rel = pd.DataFrame(index=event_features.index)
    
    # Merge customer aggregates
    cust_data = customer_agg.loc[customer_ids].reset_index(drop=True)
    
    # Сумма: отклонение от среднего клиента (z-score)
    if 'train_amt_mean' in cust_data.columns and 'train_amt_std' in cust_data.columns:
        rel['amt_zscore'] = (event_features['operaton_amt'].values - cust_data['train_amt_mean'].values) / (cust_data['train_amt_std'].values + 1e-8)
        rel['amt_ratio_to_mean'] = event_features['operaton_amt'].values / (cust_data['train_amt_mean'].values + 1e-8)
        rel['amt_ratio_to_max'] = event_features['operaton_amt'].values / (cust_data['train_amt_max'].values + 1e-8)
    
    # Ночная операция для клиента с низкой ночной активностью
    if 'train_night_ratio' in cust_data.columns:
        rel['unusual_night'] = event_features['is_night'].values * (1 - cust_data['train_night_ratio'].values)
    
    return rel


def build_all_features(data_dir, use_pretrain=True):
    """
    Полный pipeline: читаем данные чанками, строим фичи.
    """
    print("=" * 60)
    print("FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    
    # ====== Step 1: Агрегатные фичи из pretrain ======
    pretrain_agg = None
    if use_pretrain:
        print("\n[1/5] Processing pretrain data...")
        pretrain_parts = []
        for i in range(1, 4):
            print(f"  Reading pretrain_part_{i}.parquet...")
            df = pd.read_parquet(f'{data_dir}/pretrain_part_{i}.parquet')
            agg = process_chunk_features(df, prefix='pretrain_')
            pretrain_parts.append(agg)
            del df; gc.collect()
        pretrain_agg = pd.concat(pretrain_parts)
        del pretrain_parts; gc.collect()
        print(f"  Pretrain features shape: {pretrain_agg.shape}")
    
    # ====== Step 2: Агрегатные фичи из train ======
    print("\n[2/5] Processing train data...")
    train_parts = []
    train_raw_parts = []
    for i in range(1, 4):
        print(f"  Reading train_part_{i}.parquet...")
        df = pd.read_parquet(f'{data_dir}/train_part_{i}.parquet')
        agg = process_chunk_features(df, prefix='train_')
        train_parts.append(agg)
        train_raw_parts.append(df)
    train_agg = pd.concat(train_parts)
    del train_parts; gc.collect()
    print(f"  Train aggregate features shape: {train_agg.shape}")
    
    # Объединяем агрегатные фичи
    if pretrain_agg is not None:
        customer_agg = pretrain_agg.join(train_agg, how='outer')
        del pretrain_agg; gc.collect()
    else:
        customer_agg = train_agg
    del train_agg; gc.collect()
    customer_agg = customer_agg.fillna(0)
    print(f"  Combined customer features shape: {customer_agg.shape}")
    
    # ====== Step 3: Labels ======
    print("\n[3/5] Processing labels...")
    labels = pd.read_parquet(f'{data_dir}/train_labels.parquet')
    print(f"  Labels: {len(labels)} events, target=1: {(labels.target==1).sum()}, target=0: {(labels.target==0).sum()}")
    
    # Join labels with train data to get events
    train_all = pd.concat(train_raw_parts, ignore_index=True)
    del train_raw_parts; gc.collect()
    
    # ВСЕ train операции: помечаем label
    train_all = train_all.merge(labels[['event_id', 'target']], on='event_id', how='left')
    
    # target: 1 = неподтверждённая (🔴), 0 = подтверждённая (🟡), NaN = зелёная (🟢)
    # Для бинарной классификации: 🔴=1, 🟡+🟢=0
    train_all['target'] = train_all['target'].fillna(-1)  # -1 = зелёная
    
    # Берём только помеченные + сэмпл зелёных для обучения
    labeled = train_all[train_all['target'].isin([0, 1])].copy()
    
    # Добавляем сэмпл зелёных как негативный класс
    green = train_all[train_all['target'] == -1].sample(
        n=min(200_000, (train_all['target'] == -1).sum()), 
        random_state=42
    ).copy()
    green['target'] = 0  # зелёные = негативный класс
    
    train_labeled = pd.concat([labeled, green], ignore_index=True)
    del train_all, labeled, green; gc.collect()
    
    print(f"  Training set: {len(train_labeled)} events")
    print(f"  Target distribution: 1={int((train_labeled.target==1).sum())}, 0={int((train_labeled.target==0).sum())}")
    
    # ====== Step 4: Event-level features ======
    print("\n[4/5] Building event-level features...")
    train_event_features = build_event_level_features(train_labeled)
    
    # Relative features
    customer_ids = train_labeled['customer_id'].values
    valid_customers = customer_agg.index.intersection(pd.Index(customer_ids).unique())
    # Для тех у кого нет агрегатов - заполним нулями
    missing = set(customer_ids) - set(customer_agg.index)
    if missing:
        missing_df = pd.DataFrame(0, index=list(missing), columns=customer_agg.columns)
        customer_agg = pd.concat([customer_agg, missing_df])
    
    rel_features = build_relative_features(
        train_event_features, customer_agg, customer_ids
    )
    
    # Merge all
    X_train = pd.concat([train_event_features, rel_features], axis=1)
    # Add customer-level features
    cust_feat_for_merge = customer_agg.loc[customer_ids].reset_index(drop=True)
    X_train = pd.concat([X_train, cust_feat_for_merge], axis=1)
    
    y_train = train_labeled['target'].values.astype(int)
    train_event_ids = train_labeled['event_id'].values
    train_customer_ids = train_labeled['customer_id'].values
    
    del train_labeled, train_event_features, rel_features, cust_feat_for_merge
    gc.collect()
    
    print(f"  X_train shape: {X_train.shape}")
    
    # ====== Step 5: Test features ======
    print("\n[5/5] Building test features...")
    
    # Pretest aggregates
    print("  Reading pretest.parquet...")
    pretest = pd.read_parquet(f'{data_dir}/pretest.parquet')
    pretest_agg = process_chunk_features(pretest, prefix='pretest_')
    customer_agg = customer_agg.join(pretest_agg, how='outer').fillna(0)
    del pretest_agg; gc.collect()
    
    # Test data
    print("  Reading test.parquet...")
    test = pd.read_parquet(f'{data_dir}/test.parquet')
    test_event_features = build_event_level_features(test)
    
    test_customer_ids = test['customer_id'].values
    missing_test = set(test_customer_ids) - set(customer_agg.index)
    if missing_test:
        missing_df = pd.DataFrame(0, index=list(missing_test), columns=customer_agg.columns)
        customer_agg = pd.concat([customer_agg, missing_df])
    
    test_rel = build_relative_features(test_event_features, customer_agg, test_customer_ids)
    X_test = pd.concat([test_event_features, test_rel], axis=1)
    cust_feat_test = customer_agg.loc[test_customer_ids].reset_index(drop=True)
    
    # Ensure same columns
    for col in X_train.columns:
        if col not in cust_feat_test.columns and col not in X_test.columns:
            X_test[col] = 0
    
    X_test = pd.concat([X_test, cust_feat_test], axis=1)
    
    # Align columns
    common_cols = [c for c in X_train.columns if c in X_test.columns]
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    test_event_ids = test['event_id'].values
    
    del test, test_event_features, test_rel, cust_feat_test, pretest
    gc.collect()
    
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Features: {len(common_cols)}")
    print("\nFeature engineering complete!")
    
    return X_train, y_train, X_test, test_event_ids, train_customer_ids, train_event_ids


if __name__ == '__main__':
    X_train, y_train, X_test, test_event_ids, train_customer_ids, train_event_ids = build_all_features(
        './', use_pretrain=False  # Для быстрого теста без pretrain
    )
    print(X_train.head())

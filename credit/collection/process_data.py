import pandas as pd
import csv

df_sour = pd.read_csv('temp_dm_v_tr_duebill_mx.csv', encoding='utf-8', usecols=[
    'uid', 'acct_no', 'buss_amt', 'dlay_amt', 'loan_use', 'pay_type', 'pay_freq', 'vouch_type'
])
file = open('dm_v_tr_duebill_mx.csv', mode='a+', newline='')
writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow([
    'uid', 'buss_amt', 'dlay_amt', 'loan_use', 'pay_type', 'pay_freq', 'vouch_type'
])
uid_list = df_sour['uid'].unique()
num_columns = []
columns = []
for col in df_sour.columns:
    columns.append(col)
dtypes = df_sour.dtypes
for i in range(0, len(columns)):
    if dtypes[i] != 'object':
        num_columns.append(columns[i])
columns.remove('uid')
columns.remove('acct_no')
for uid in uid_list:
    row = [uid]
    df = df_sour.loc[df_sour['uid'] == uid]
    for col in columns:
        if col in num_columns:
            row.append(df[col].sum())
        else:
            counts = df[col].value_counts()
            if counts.empty:
                row.append(None)
            else:
                row.append(counts.idxmax())
    writer.writerow(row)

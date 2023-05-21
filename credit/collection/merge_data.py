import pandas as pd

contract_data = pd.read_csv('dm_v_tr_contract_mx.csv', encoding='utf-8')
asset_data = pd.read_csv('pri_cust_asset_info.csv', encoding='utf-8')
base_data = pd.read_csv('pri_cust_base_info.csv', encoding='utf-8')
liab_acct_info = pd.read_csv('pri_cust_liab_acct_info.csv', encoding='utf-8')
liab_info = pd.read_csv('pri_cust_liab_info.csv', encoding='utf-8')
duebill_info = pd.read_csv('dm_v_tr_duebill_mx.csv', encoding='utf-8')
credit_data = pd.read_csv('pri_credit_info_known.csv', encoding='utf-8')

merge_data = pd.merge(credit_data, contract_data, how='left', on='uid')
merge_data = pd.merge(merge_data, asset_data, how='left', on='uid')
merge_data = pd.merge(merge_data, base_data, how='left', on='uid')
merge_data = pd.merge(merge_data, liab_acct_info, how='left', on='uid')
merge_data = pd.merge(merge_data, liab_info, how='left', on='uid')
merge_data = pd.merge(merge_data, duebill_info, how='left', on='uid')

merge_data.to_csv('../csv4/train.csv', index=False)

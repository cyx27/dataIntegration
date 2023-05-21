import csv
import clickhouse_driver

# 连接 ClickHouse 数据库
conn = clickhouse_driver.connect(
    host='101.43.99.84',
    port=9001,
    database='dm',
)

with open('csv_process/pri_cust_liab_acct_info.csv_process', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv_process.QUOTE_MINIMAL)
    writer.writerow(
        ['belong_org', 'exam_org', 'cust_no', 'loan_cust_no', 'cust_name', 'uid', 'acct_no', 'begin_date', 'matu_date',
         'settle_date', 'subject_no', 'prod_type', 'buss_type', 'buss_type_name', 'loan_type', 'float_tpename',
         'loan_amt', 'loan_bal', 'loan_mgr_no', 'loan_mgr_name', 'mgr_phone', 'vouch_type', 'putout_channel',
         'next_repay_date', 'is_mortgage', 'is_online', 'is_extend', 'extend_date', 'ext_matu_date', 'repay_type',
         'term_mth', 'five_class', 'overdue_class', 'overdue_flag', 'owed_int_flag', 'contract_no', 'credit_amt',
         'credit_begin_date', 'credit_matu_date', 'frst_intr', 'actu_intr', 'loan_mob_phone', 'loan_use',
         'inte_settle_type', 'bankacct', 'defect_type', 'owed_int_in', 'owed_int_out', 'delay_bal', 'industr_type',
         'industr_type_name', 'acct_sts', 'arti_ctrt_no', 'ext_ctrt_no', 'flst_teller_no', 'attract_no', 'attract_name',
         'loan_use_add', 'loan_use_add', 'putout_acct', 'is_book_acct', 'book_acct_buss', 'book_acct_amt',
         'sub_buss_type',
         'pro_char', 'pro_char_ori', 'pay_type', 'grntr_name', 'grntr_cert_no', 'guar_no', 'guar_right_no', 'guar_name',
         'guar_amount', 'guar_add', 'guar_eva_value', 'guar_con_value', 'guar_reg_date', 'guar_matu_date', 'etl_dt'])
    # 执行查询
    cursor = conn.cursor()
    cursor.execute('select * from pri_cust_liab_acct_info')
    result = cursor.fetchall()
    for row in result:
        print(row)
    for item in result:
        writer.writerow(item)
    conn.close()
import csv

import clickhouse_driver

# 连接 ClickHouse 数据库
conn = clickhouse_driver.connect(
    host='101.43.99.84',
    port=9001,
    database='dm',
)

cursor = conn.cursor()

with open('../csv/tr_mx/dsf.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt from dm_v_tr_dsf_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)


with open('../csv/tr_mx/etc.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt_fen from dm_v_tr_etc_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)


with open('../csv/tr_mx/grwy.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt from dm_v_tr_grwy_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)


with open('../csv/tr_mx/gzdf.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt from dm_v_tr_gzdf_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)


with open('../csv/tr_mx/sa.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt from dm_v_tr_sa_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)

with open('../csv/tr_mx/sbyb.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt_fen from dm_v_tr_sbyb_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)

with open('../csv/tr_mx/sdrq.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt_fen from dm_v_tr_sdrq_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)


with open('../csv/tr_mx/shop.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt from dm_v_tr_shop_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)


with open('../csv/tr_mx/sjyh.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'tran_date', 'tran_amt'])
    # 执行查询
    cursor.execute('select uid,tran_date,tran_amt from dm_v_tr_sjyh_mx')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)


# 关闭连接
conn.close()

import csv

import clickhouse_driver

# 连接 ClickHouse 数据库
conn = clickhouse_driver.connect(
    host='101.43.99.84',
    port=9001,
    database='dm',
)

cursor = conn.cursor()

with open('../csv/pri_cust_base_info.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['uid', 'is_shareholder', 'is_black', 'is_contact', 'is_mgr_dep'])
    # 执行查询
    cursor.execute('select uid,is_shareholder,is_black, is_contact, is_mgr_dep from pri_cust_base_info')
    result = cursor.fetchall()
    for row in result:
        print(row)
        writer.writerow(row)

conn.close()

import pandas
df = pandas.read_csv("../csv/pri_cust_base_info.csv")
df["is_mgr_dep"].fillna('N', inplace=True)
df.to_csv("../csv/pri_cust_base_info.csv", index=False)
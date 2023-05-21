import clickhouse_driver
import csv

# 连接 ClickHouse 数据库
conn = clickhouse_driver.connect(
    host='101.43.99.84',
    port=9001,
    database='dm',
)

all = set([])

with open('prim_cust_asset_info.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for start in range(0, 344130, 1000):
        # 执行查询
        cursor = conn.cursor()
        cursor.execute('select * from pri_cust_asset_info limit ' + str(start) + ',1000')
        result = cursor.fetchall()
        for row in result:
            print(row)


        for item in result:
            if item[2] is None or item[6] is None or item[8] is None:
                continue
            temp = []
            uid = str(item[2])
            if uid in all:
                continue
            all.add(uid)
            avg_month = float(item[6])
            avg_year = float(item[8])
            if avg_month + avg_year <= float(0):
                continue
            temp.append(uid)
            temp.append(avg_month)
            temp.append(avg_year)
            writer.writerow(temp)

    # 关闭连接
    conn.close()

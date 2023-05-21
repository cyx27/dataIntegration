import clickhouse_driver
import csv

# 连接 ClickHouse 数据库
conn = clickhouse_driver.connect(
    host='101.43.99.84',
    port=9001,
    database='dm',
)

with open('pri_star_info_marked.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # 执行查询
    cursor = conn.cursor()
    cursor.execute('select * from pri_star_info')
    result = cursor.fetchall()
    for item in result:
        if not int(item[1]) == -1:
            writer.writerow(item)


with open('pri_star_info_unmarked.csv', mode='a+', newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # 执行查询
    cursor = conn.cursor()
    cursor.execute('select * from pri_star_info')
    result = cursor.fetchall()
    for item in result:
        if int(item[1]) == -1:
            writer.writerow(item)

conn.close()
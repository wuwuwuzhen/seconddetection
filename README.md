# docker
```shell
docker exec -it 69 /bin/zsh
ps -ef | grep python
cd data/code/hf_bus
pkill -f python
kill -9 $(pgrep gunicorn)
tail -f -n 100 logs/hf_bus.log
nohup gunicorn --workers 1 --threads 1  --timeout 3600 --bind 0.0.0.0:5000 main:app --log-level debug --access-logfile - --error-logfile - &
```
# 生成patch
```shell
python ./get_patch.py
git am ./patch/*.patch
```
```
# curl 
```sehll
curl --location 'http://127.0.0.1:5000/seconddetection/' \
--header 'Content-Type: application/json' \
--data '[
    {
        "alarm_end_time": "2024-01-19 23:25:30",
        "exception_name": "车道偏离（右）预警",
        "picture_url": [
            "https://10.2.137.173:443/ngx/proxy?i=aHR0cDovLzEwLjIuMTM3LjEyOTo2MTIwL3BpYz89ZDQxaWQ3NGUqNjA1aTA5YS1jMDc3MTBlNT10MWkybSo9cDFwM2k9ZDFzKmk5ZDA9KjdiMWkwZGU3KjYyY2M5YTg0Yi1hMmVlMTYzLTFmNzBzNi1kMHo5NzlkOD1jNg=="
        ],
        "alarm_begin_time": "2024-01-19 23:25:30",
        "exception_type": "5",
        "plate": "皖A20685D",
        "id": 240119232600001,
        "day": "2024-01-19"
    },
    {
        "alarm_end_time": "2024-01-19 23:25:05",
        "exception_name": "车道偏离（左）预警",
        "picture_url": [
            "https://10.2.137.173:443/ngx/proxy?i=aHR0cDovLzEwLjIuMTM3LjEyOTo2MTIwL3BpYz89ZDkxaWQ3OWUqNjA1aTA5YS1jNTc3MTBlNT10MWk1bSo9cDNwOWk9ZDFzKmk0ZDY9KjZiMmkwZGU3KjYyY2M5YTg0Yi1hMmVlMTYwLTFmNzBzNi1kMHoxNzlkNz1jMg=="
        ],
        "alarm_begin_time": "2024-01-19 23:25:05",
        "exception_type": "3",
        "plate": "皖A06039D",
        "id": 240119232600003,
        "day": "2024-01-19"
    },
    {
        "alarm_end_time": "2024-01-19 23:24:47",
        "exception_name": "车道偏离（左）预警",
        "picture_url": [
            "https://10.2.137.173:443/ngx/proxy?i=aHR0cDovLzEwLjIuMTM3LjEyOTo2MTIwL3BpYz89ZDUxaWQ3OWUqNjA1aTA4YS1jNzc3MTBlNT10MWk0bSo9cDNwOWk9ZDFzKmk0ZDY9KjZiMmkwZGU3KjYyY2M5YTg0Yi1hMmVlMTY4LTFmNzBzNi1kMHoxNzlkNz1jNg=="
        ],
        "alarm_begin_time": "2024-01-19 23:24:47",
        "exception_type": "3",
        "plate": "皖A06039D",
        "id": 240119232600004,
        "day": "2024-01-19"
    },
    {
        "alarm_end_time": "2024-01-19 23:24:42",
        "exception_name": "车道偏离（左）预警",
        "picture_url": [
            "https://10.2.137.173:443/ngx/proxy?i=aHR0cDovLzEwLjIuMTM3LjEyOTo2MTIwL3BpYz89ZDAxaWQ3MmUqNjA1aTA4YS1jMjc3MTBlNT10MWk0bSo9cDNwNWk9ZDFzKmk3ZDQ9KjViOWkwZGU3KjYyY2M5YTg0Yi1hMmVlMTY4LTFmNzBzNi1kMHowNzlkOD1jOQ=="
        ],
        "alarm_begin_time": "2024-01-19 23:24:42",
        "exception_type": "3",
        "plate": "皖A05762D",
        "id": 240119232600005,
        "day": "2024-01-19"
    },
    {
        "alarm_end_time": "2024-01-19 23:24:41",
        "exception_name": "车道偏离（右）预警",
        "picture_url": [
            "https://10.2.137.173:443/ngx/proxy?i=aHR0cDovLzEwLjIuMTM3LjEyOTo2MTIwL3BpYz89ZDAxaWQ3NWUqNjA1aTA4YS1jMTc3MTBlNT10MWk1bSo9cDFwN2k9ZDFzKmkxZDg9KjdiMGkwZGU3KjYyY2M5YTg0Yi1hMmVlMTY4LTFmNzBzNi1kMHo4NzlkMD1jMA=="
        ],
        "alarm_begin_time": "2024-01-19 23:24:41",
        "exception_type": "5",
        "plate": "皖A20685D",
        "id": 240119232600006,
        "day": "2024-01-19"
    }
]'

```
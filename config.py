import os

root_path = os.path.dirname(__file__)
photo_path = os.path.join(root_path, 'picture')
video_path = os.path.join(root_path, 'video')
csv_path = os.path.join(root_path, 'csv')
log_dir = os.path.join(root_path, 'logs')

vit_l_14_path = os.path.join(root_path, 'ViT-L-14.pt')
rn_50_x64_path = os.path.join(root_path, 'RN50x64.pt')

ak_secret = "c9be232e30284969b72ac5fac4135113"
resp_url = 'http://10.2.137.136:9202/alarm/filter/receive'

filter_exception_name = ["疲劳驾驶", "疲劳驾驶预警", "分神提醒", "分神提醒预警"]
import json
p = r"D:\AdverseWeather\datasets\cadcd_small\0002\3d_ann.json"
with open(p,'r',encoding='utf-8') as f:
    data = json.load(f)
print("top type:", type(data))
if isinstance(data, dict):
    print("top keys:", list(data.keys())[:30])
    # 尝试找一个“第一条标注记录”
    for k in data.keys():
        v = data[k]
        if isinstance(v, list) and len(v)>0:
            print("example list key:", k, "len", len(v))
            print("first item keys:", list(v[0].keys())[:30] if isinstance(v[0], dict) else type(v[0]))
            break
elif isinstance(data, list) and data:
    print("list len:", len(data))
    print("first item keys:", list(data[0].keys())[:30] if isinstance(data[0], dict) else type(data[0]))

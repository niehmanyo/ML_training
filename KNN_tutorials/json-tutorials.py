import json

a = {
    "name":"wenyu",
    "age":32
}
y=json.dumps(a)
print(y.__class__)
print(a.__class__)
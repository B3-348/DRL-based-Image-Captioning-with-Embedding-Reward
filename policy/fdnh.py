import pandas as pd
a=[{'a': 1,  'b': 2,  'c': 3},
   {'a': 11, 'b': 132, 'c': 13},
   {'a': 21, 'b': 22, 'c': 23},
   {'a': 31, 'b': 32, 'c': 33}]
b={
    'a' : [1, 2, 3, 4],
    'b' :[11,12,13,14],
    'c' :[21,22,23,24],
    'd' :[31,32,33,34],

}


data = pd.DataFrame.from_dict(a)
da=pd.DataFrame.from_dict(a)
del data['a']
del da['a']
data.sort_values(by='b',inplace=True)
print(data)
data = data.reset_index(drop=True)
print(data)
da.sort_values(by='b',inplace=False)
print(da)

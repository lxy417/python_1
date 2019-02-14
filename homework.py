import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#pandas series
s=pd.Series([1,3,5,np.nan,6,8])
print(s)
print("-------------"*10)
s1=pd.Series(np.random.rand(4),index=list("abcd"))
print(s1)
d={'刘星宇':1.,'aaa':2.}
print(pd.Series(d))


#pandas  dataframe
datas=pd.date_range('20170101',periods=7)
print(datas)
print("-------------"*10)
df=pd.DataFrame(np.random.rand(7,4),index=datas,columns=list('ABCD'))
print(df)
print("-------------"*10)
print(df.head(1))
print("-------------"*10)
print(df.tail(2))
print("-------------"*10)
df2 = pd.DataFrame({ 'A' : 1.,
                     'B' : pd.Timestamp('20170102'),
                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),
                     'D' : np.array([3] * 4,dtype='int32'),
                     'E' : pd.Categorical(["test","train","test","train"]),
                     'F' : 'foo' })
print(df2)
print("-------------"*10)
print(df2.dtypes)
print("-------------"*10)
print("index is :" )
print(df.index)
print("-------------"*10)
print("columns is :" )
print(df.columns)
print("-------------"*10)
print("values is :" )
print(df.values)
print("-------------"*10)
print(df2.describe())
print("-------------"*10)
print(df)
dft=df.T
print(df.T)
print("-------------"*10)
d2={'one':s1,'two':s1}
print(d2)
print("-------------"*10)
df3=pd.DataFrame(d2)
print(pd.DataFrame(d2))
print("-------------"*10)

#Pandas plot
s=pd.Series(np.random.rand(4),index=list("abcd"),name='series')
s.plot.pie(figsize=(6,6))
df=pd.DataFrame(np.random.rand(7,4),index=list("abcdefg"),columns=list("abcd"))
df.plot.bar()
df2=pd.DataFrame(np.random.rand(7,5),columns=list("abcde"))
df2.plot.box()
df3=pd.DataFrame(np.random.rand(100,4),columns=list("abcd"))
df3.plot.scatter(x='a',y='b')
plt.show()




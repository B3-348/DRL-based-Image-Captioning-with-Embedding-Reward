import tensorflow as tf
a=[
    [1,2,23,4353,57],
    [34,546,46,67,78],
    [1223,34,35,65,7]

]
b=[
    [1,2,23333,4353,57],
    [34,546,4446,67,78],
    [1223,34,3445,65,7]

]
c=[
    [1,22222,23,4353,57],
    [32224,546,46,67,78],
    [1223,34,35,65,72222]

]
hh=[]
with tf.Session() as sess:
    d=tf.argmax(a,1)
    e=tf.argmax(b,1)
    f=tf.argmax(c,1)
    hh.append(d)
    hh.append(e)
    hh.append(f)
    gg=tf.transpose(tf.stack(hh),(1,0))
    print(sess.run(gg))
    print(sess.run(len(gg)))
    if gg.ndim == 1:
        print("0000")
    else:
        print("nnn")



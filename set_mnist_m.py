import os

with open('../svhn2mnist_m/mnist_m_test_labels.txt', 'rb') as f:
    while True:
        s = f.readline()
        s = str(s)
        name = s.split(' ')[0]
        name = 'test01/' + name
        label = s.split(' ')[1]

        with open('../svhn2mnist_m/test_label_m.txt', 'a') as h:
            temp = name + ' ' + str(label)
            h.write(temp)
            h.write('\n')

import numpy as np
import struct
import matplotlib.pyplot as plt
# 每层节点数
d0size = 784
d1size = 16
d2size = 16
d3size = 10


class NeuralNetwork:
    def __init__(self):
        self.K = d3size  # 目标空间维度
        # learning rate
        self.step_size = 0.01
        self.BATCH = 4000
        self.NTRAIN = 60000
        self.NTEST = 10000

        self.train_images_list = np.zeros((self.NTRAIN, d0size))  # 读入60000张图像
        self.train_labels_list = np.zeros((self.NTRAIN, 1))  # 读入60000个标记

        self.test_images_list = np.zeros((self.NTEST, d0size))  # 读入10000张图像
        self.test_labels_list = np.zeros((self.NTEST, 1))  # 读入10000个标记
        # 文件
        self.train_image_file = "train-images.idx3-ubyte"
        self.train_label_file = "train-labels.idx1-ubyte"

        self.test_image_file = "t10k-images.idx3-ubyte"
        self.test_label_file = "t10k-labels.idx1-ubyte"

        self.lost_list = []
        self.init_net()
        return

    def read_train(self):
        self.read_train_images(self.train_image_file)
        self.read_train_labels(self.train_label_file)

        self.train_data = np.append(self.train_images_list, self.train_labels_list, axis=1)

    def read_test(self):

        self.read_test_images(self.test_image_file)
        self.read_test_labels(self.test_label_file)

    def read_all(self):
        self.read_train_images(self.train_image_file)
        self.read_train_labels(self.train_label_file)

        self.train_data = np.append(self.train_images_list, self.train_labels_list, axis=1)

        # self.read_test_image()
        # self.read_test_label()
        self.read_test_images(self.test_image_file)
        self.read_test_labels(self.test_label_file)

        print(self.train_images_list.T)

    def train(self, s=100):
        self.read_train()
        for i in range(s):
            np.random.shuffle(self.train_data)
            train_img_batch = self.train_data[:self.BATCH, :-1]
            train_lab_batch = self.train_data[:self.BATCH, -1:]
            print('训练次数; ', i+1)
            self.train_net(train_img_batch, train_lab_batch)

    def train_net(self, train_images_batch, train_labels_batch):
        # 训练数量 也可以等于self.BATCH
        example_num = train_images_batch.shape[0]
        # ai = ReLU(a<i-1> * wi + bi)
        activition1 = np.maximum(0, np.matmul(train_images_batch, self.weight1) + self.biases1)
        activition2 = np.maximum(0, np.matmul(activition1, self.weight2) + self.biases2)
        scores = np.maximum(0, np.matmul(activition2, self.weight3) + self.biases3)
        # 2000行 1列
        # 2000行 10列 每个元素的得分
        scores_e = np.exp(scores)
        scores_e_sum = np.sum(scores_e, axis=1, keepdims=True)
        probs = scores_e / scores_e_sum

        # backpropagation

        dscores = np.zeros((example_num, self.K))
        for i in range(example_num):
            dscores[ i ][ : ] = probs[ i ][ : ]
            dscores[ i ][int(train_labels_batch[ i ])] -= 1
        dscores /= example_num
        loss = np.sum(dscores*dscores)
        print(loss)
        dw3 = np.dot(activition2.T, dscores)
        db3 = np.sum(dscores, axis=0, keepdims=True)

        da2 = np.dot(dscores, self.weight3.T)
        da2[activition2 <= 0] = 0

        dw2 = np.dot(activition1.T, da2)
        db2 = np.sum(da2, axis=0, keepdims=True)

        da1 = np.dot(da2, self.weight2.T)
        da1[activition1 <= 0] = 0

        dw1 = np.dot(train_images_batch.T, da1)
        db1 = np.sum(da1, axis=0, keepdims=True)


        self.weight3 += -self.step_size * dw3
        self.weight2 += -self.step_size * dw2
        self.weight1 += -self.step_size * dw1

        self.biases3 += -self.step_size * db3
        self.biases2 += -self.step_size * db2
        self.biases1 += -self.step_size * db1

        return

    def test_net(self):
        self.read_test()
        activition1 = np.maximum(0, np.matmul(self.test_images_list, self.weight1) + self.biases1)
        activition2 = np.maximum(0, np.matmul(activition1, self.weight2) + self.biases2)
        activition3 = np.maximum(0, np.matmul(activition2, self.weight3) + self.biases3)

        answer = np.argmax(activition3, axis=1)
        answer = np.reshape(answer, (10000, 1))
        scores = np.mean(answer == self.test_labels_list)
        print('识别准确率; ',scores)
        return

    def init_net(self):

        self.weight1 = 0.01 * np.random.randn(d0size, d1size)
        self.biases1 = 0.01 * np.random.randn(1, d1size)

        self.weight2 = 0.01 * np.random.randn(d1size, d2size)
        self.biases2 = 0.01 * np.random.randn(1, d2size)

        self.weight3 = 0.01 * np.random.randn(d2size, self.K)
        self.biases3 = 0.01 * np.random.randn(1, self.K)

    def read_train_images(self,filename):
        binfile = open(filename, 'rb')
        buf = binfile.read()
        index = 0
        magic, self.train_img_num, self.numRows, self.numColums = struct.unpack_from('>IIII', buf, index)
        print(magic, ' ', self.train_img_num, ' ', self.numRows, ' ', self.numColums)
        index += struct.calcsize('>IIII')
        for i in range(self.train_img_num):
            im = struct.unpack_from('>784B', buf, index)
            index += struct.calcsize('>784B')
            im = np.array(im)
            im = im.reshape(1, d0size)
            self.train_images_list[ i , : ] = im

            # plt.imshow(im, cmap='binary')  # 黑白显示
            # plt.show()

    def read_train_labels(self,filename):
        binfile = open(filename, 'rb')
        index = 0
        buf = binfile.read()
        binfile.close()
        magic, self.train_label_num = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')
        for i in range(self.train_label_num):
            label_item = int(struct.unpack_from('>B', buf, index)[0])
            self.train_labels_list[ i , : ] = label_item
            index += struct.calcsize('>B')

    def read_test_images(self, filename):
        binfile = open(filename, 'rb')
        buf = binfile.read()
        index = 0
        magic, self.test_img_num, self.numRows, self.numColums = struct.unpack_from('>IIII', buf, index)
        print(magic, ' ', self.test_img_num, ' ', self.numRows, ' ', self.numColums)
        index += struct.calcsize('>IIII')
        for i in range(self.test_img_num):
            im = struct.unpack_from('>784B', buf, index)
            index += struct.calcsize('>784B')
            im = np.array(im)
            im = im.reshape(1, d0size)
            self.test_images_list[i, :] = im

    def read_test_labels(self,filename):
        binfile = open(filename, 'rb')
        index = 0
        buf = binfile.read()
        binfile.close()

        magic, self.test_label_num = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')

        for i in range(self.test_label_num):
            label_item = int(struct.unpack_from('>B', buf, index)[0])
            self.test_labels_list[i, :] = label_item
            index += struct.calcsize('>B')

    def read_train_image(self):
        index = 0
        binfile = open(self.train_image_file, 'rb')
        buf = binfile.read()
        magic, self.num_Trian_Images, self.numRows, self.numColumns = struct.unpack_from('>IIII', buf, index)
        index += struct.calcsize('>IIII')
        for i in range(self.NTRAIN):
            im = struct.unpack_from('>784B', buf, index)
            index += struct.calcsize('>784B')
            im = np.array(im)
            self.train_image_list[i, :] = im.reshape(1,784)
        print(len(self.train_image_list), ' train images in')

    def read_train_label(self):
        index = 0
        binfile = open(self.train_label_file, 'rb')
        buf = binfile.read()
        magic, self.num_Trian_Labels = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')
        for i in range(self.NTRAIN):
            lb = int(struct.unpack_from('>B', buf, index)[0])
            index += struct.calcsize('>B')
            self.train_label_list[i, 0] = lb
        print(len(self.train_label_list), ' train labels in')

    def read_test_image(self):
        index = 0
        binfile = open(self.test_image_file, 'rb')
        buf = binfile.read()
        magic, self.num_Test_Images, self.numRows, self.numColumns = struct.unpack_from('>IIII', buf, index)
        index += struct.calcsize('>IIII')
        for i in range(self.NTEST):
            im = struct.unpack_from('>784B', buf, index)
            index += struct.calcsize('>B')
            self.test_image_list[i, :] = np.mat(im).reshape(1, d0size)
        print(len(self.test_image_list), ' test images in')

    def read_test_label(self):
        index = 0
        binfile = open(self.test_label_file, 'rb')
        buf = binfile.read()
        magic, self.num_Test_Labels = struct.unpack_from('>II', buf, index)
        index += struct.calcsize('>II')
        for i in range(self.NTEST):
            lb = int(struct.unpack_from('>B', buf, index)[0])
            index += struct.calcsize('>B')
            self.test_label_list[i, :] = lb
        print(len(self.test_label_list), ' test labels in')

    def use_net(self):
        self.read_test()
        # for i in range(self.NTEST):
        x = 6
        y = 10
        for i in range(x*y):
            im = self.test_images_list[i]
            im = np.array(im)
            im = im.reshape(1,d0size)
            # self.recognize(im)
            activition1 = np.maximum(0, np.matmul(im, self.weight1) + self.biases1)
            activition2 = np.maximum(0, np.matmul(activition1, self.weight2) + self.biases2)
            scores = np.maximum(0, np.matmul(activition2, self.weight3) + self.biases3)
            answer = np.argmax(scores)
            plt.subplot(x,y,i+1)
            plt.imshow(im.reshape(28, 28))
            plt.title('answer'+str(answer))
            plt.axis('off')
        plt.show()

    def recognize(self ,image):
        activition1 = np.maximum(0, np.matmul(image, self.weight1) + self.biases1)
        activition2 = np.maximum(0, np.matmul(activition1, self.weight2) + self.biases2)
        scores = np.maximum(0, np.matmul(activition2, self.weight3) + self.biases3)
        answer = np.argmax(scores)
        image = image.reshape(28,28)
        print('识别结果： ',answer)
        plt.ion()
        plt.imshow(image, cmap='binary')  # 黑白显示
        plt.show()
        plt.pause(1)
        plt.close()

    def save(self):
        np.save("weight1.npy", self.weight1)
        np.save("weight2.npy", self.weight2)
        np.save("weight3.npy", self.weight3)
        np.save("bias1.npy", self.biases1)
        np.save("bias2.npy", self.biases2)
        np.save("bias3.npy", self.biases3)

    def read(self):
        self.weight1 = np.load("weight1.npy")
        self.weight2 = np.load("weight2.npy")
        self.weight3 = np.load("weight3.npy")
        self.biases1 = np.load("bias1.npy")
        self.biases2 = np.load("bias2.npy")
        self.biases3 = np.load("bias3.npy")

    def show(self):
        plt.figure()
        for i in range(d1size):
            plt.subplot(4,4,i+1)
            w = self.weight1[:, i].reshape(28, 28)
            plt.imshow(w)
            plt.axis('off')
        plt.show()


def guide(net):
    ch = ''
    while ch!='0':
        ch = input("1.read_all\n2.train\n3.test\n4.save\n5.read\n6.show\n7.use\n8.train_c\n0.exit\n:")
        if ch == '1':
            net.read_all()
        elif ch == '2':
            net.train()
        elif ch == '3':
            net.test_net()
        elif ch == '4':
            net.save()
        elif ch == '5':
            net.read()
        elif ch == '6':
            net.show()
        elif ch == '7':
            net.use_net()
        elif ch == '8':
            i = input("训练次数")
            net.train(s=i)
        else:
            print("没有这样的函数")


def main():
    net = NeuralNetwork()
    guide(net)


if __name__ == '__main__':
    main()

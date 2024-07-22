import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    # load_data函數接受一個叫做data_dir的參數
    # Return的值都包含images(array) and labels(integer)；分別是圖案本身的描述以及它的種類
    # array是width X height X 3(RGB)

    # 初始化圖像和標籤列表
    images = []
    labels = []

    # 遍歷每一個類別目錄
    for category in range(NUM_CATEGORIES):
        #把不同文件的路徑拼接
        category_path = os.path.join(data_dir, str(category))

        # 確保目錄存在
        if not os.path.isdir(category_path):
            continue

        # 遍歷每一個圖像文件
        for filename in os.listdir(category_path):
            # 構建完整的圖像路徑
            img_path = os.path.join(category_path, filename)

            # 使用 OpenCV 加載圖像
            image = cv2.imread(img_path)
            if image is None:
                continue

            # 調整圖像大小
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))

            # 將圖像添加到列表中
            images.append(image)

            # 添加對應的標籤
            labels.append(category)

    return images, labels


def get_model():
    # return: 編譯好的CNN模型 (需要tensorflow和keras)
    # Convolutional Neural Network，由convolutional layer、activation function、pooling layer、fully connected layer、output layer組成
    # 可用ReLU引入非線性激活函數，Max pooling找到相對應特徵，再flatten把每個類別的output顯示出來
    # 輸出層一個類別對應一個單位，且每個類別對應一個輸出
    # 需有損失函數、優化器和評估指標

    # 創建一個順序模型
    # 每一層的輸出都是下一層的輸入
    model = tf.keras.models.Sequential()

    # 添加卷積層、池化層和其他層
    # 32個(3,3)大小的捲機核 (又叫做convolutional filter)
    # 可提取特徵、並發現圖片中類似的部分
    # Relu把負數變成0，正數維持原本 (雖然是兩種線性，但加起來就是不線性了呀)
    # 這樣非線性有助於Neural Network學習複雜的模式
    # 選擇ReLu有幾個原因:計算簡單、非線性、適合圖像識別、輸出不會被壓縮
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

    #Dropout一半的神經元來防止過擬核
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))

    # 輸出層，使用 softmax 激活函數
    model.add(tf.keras.layers.Dense(NUM_CATEGORIES, activation='softmax'))

    # 編譯模型
    # Adam是優化器；分類交叉傷的損失函數比較適合分類問題
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

if __name__ == "__main__":
    main()

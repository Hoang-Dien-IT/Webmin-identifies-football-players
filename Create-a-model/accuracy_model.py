import cv2
import os
import numpy as np
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model = load_model("soccer_face_model.h5")
folder = 'Test_data/'

categories = ["{:05d}".format(number) for number in range(60)]

accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
y_true_all = []
y_pred_all = []

for category in categories:
    imagePaths = []
    y_true = []
    y_pred = []

    for f in os.listdir(folder + category):
        imagePaths.append(folder + category + '/' + f)
        true_label = int(category)
        y_true.append(true_label)

    if not imagePaths:
        print(f"Không có ảnh trong thư mục {category}.")
    else:
        for imagePath in imagePaths:
            image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(src=image, dsize=(100, 100))
            image = np.array(image)
            image = image.reshape((1, 100, 100, 1))
            image = image / 255.0

            predictions = model.predict(image)
            predicted_label = np.argmax(predictions)
            y_pred.append(predicted_label)

        y_true_all.extend(y_true)
        y_pred_all.extend(y_pred)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

        print(f"\nMetrics for category {category}:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1-score: {f1:.2f}")

average_accuracy = np.mean(accuracy_list)
average_precision = np.mean(precision_list)
average_recall = np.mean(recall_list)
average_f1 = np.mean(f1_list)

print("\nChỉ số chung bình:")
print(f"Trung bình Accuracy: {average_accuracy:.2f}")
print(f"Trung bình Precision: {average_precision:.2f}")
print(f"Trung bình Recall: {average_recall:.2f}")
print(f"Trung bình F1-score: {average_f1:.2f}")


metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']

values = [average_accuracy, average_precision, average_recall, average_f1]

colors = ['blue', 'green', 'orange', 'red']

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=colors)

for i in range(len(metrics)):
    plt.text(i, values[i] + 0.01, f'{values[i]:.2f}', ha='center')

# Biểu đồ chỉ số đánh giá
plt.title('Average Metrics')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.show()

# Tính toán ma trận nhầm lẫn
cm = confusion_matrix(y_true_all, y_pred_all)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(15, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=categories, yticklabels=categories)
plt.xlabel('Nhãn Dự đoán', fontsize=14)
plt.ylabel('Nhãn Thực tế', fontsize=14)
plt.title('Ma trận nhầm lẫn giữa nhãn thực tế và nhãn dự đoán', fontsize=16)
plt.show()


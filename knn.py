import math

def euclidean_distance(p1, p2):
    return math.sqrt(sum((p1[i] - p2[i]) ** 2 for i in range(len(p1))))


def knn_predict(train_set, test_set, k):
    predictions = []
    train_features = [row[1:-1] for row in train_set]
    train_labels = [row[-1] for row in train_set]

    for test_row in test_set:
        test_features = test_row[1:-1]

        distances = []
        for i in range(len(train_features)):
            dist = euclidean_distance(test_features, train_features[i])
            distances.append((dist, train_labels[i]))

        distances.sort(key=lambda x: x[0])
        nearest_labels = [label for _, label in distances[:k]]

        label_counts = {}
        for label in nearest_labels:
            if label not in label_counts:
                label_counts[label] = 1
            else:
                label_counts[label] += 1
        
        most_common_label = max(label_counts, key=label_counts.get)
        predictions.append(most_common_label)

    return predictions

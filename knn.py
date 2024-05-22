import math


def euclidean_distance(point1, point2):
    distance = 0.0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return math.sqrt(distance)


def get_neighbors(train_data, test_point, k):
    distances = []
    for train_point in train_data:
        if len(test_point) != len(train_point) - 1:
            print(f"Dimensiones inconsistentes: {len(test_point)} vs {len(train_point) - 1}")
            continue
        distance = euclidean_distance(test_point, train_point[:-1])
        distances.append((train_point, distance))
    distances.sort(key=lambda x: x[1])

    neighbors = []
    for i in range(min(k, len(distances))):
        neighbors.append(distances[i][0])
    return neighbors


def predict_classification(train_data, test_point, k):
    neighbors = get_neighbors(train_data, test_point, k)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Dataset
dataset = [
    [158, 58, 'M'],
    [158, 59, 'M'],
    [158, 63, 'M'],
    [160, 59, 'M'],
    [160, 60, 'M'],
    [163, 60, 'M'],
    [163, 61, 'M'],
    [160, 64, 'M'],
    [163, 64, 'L'],
    [165, 61, 'L'],
    [165, 62, 'L'],
    [165, 65, 'L'],
    [168, 62, 'L'],
    [168, 63, 'L'],
    [168, 66, 'L'],
    [170, 63, 'L'],
    [170, 64, 'L'],
    [170, 68, 'L']
]

#  conjunto de entrenamiento y prueba
train_set = dataset[:14]
test_set = [row[:-1] for row in dataset[14:]]  # Excluyendo las etiquetas en el conjunto de prueba

# Parámetro k
k = 3

# Predicciones
predictions = []
for row in test_set:
    prediction = predict_classification(train_set, row, k)
    predictions.append(prediction)

# Resultados
actual = [row[-1] for row in dataset[14:]]
accuracy = accuracy_metric(actual, predictions)
print(f'Predicciones: {predictions}')
print(f'Actual: {actual}')
print(f'Precisión: {accuracy:.2f}%')

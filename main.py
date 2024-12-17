import re
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Загрузка текста
with open("na-dne.txt", "r", encoding="utf-8") as f:
    text = f.read()
ы
# Препроцессинг
# Очистка текста
text = re.sub(r"[^а-яА-Яa-zA-Z\s]", " ", text)  # Удаление лишних символов
text = text.lower()  # Приведение к нижнему регистру

# Токенизация
tokens = word_tokenize(text, language="russian")  # Для русского текста

# Построение словаря
vocab = list(set(tokens))
print(f"Размер словаря: {len(vocab)}")

# Обучение Skip-Gram
model = Word2Vec(sentences=[tokens], vector_size=100, window=5, min_count=1, workers=4, sg=1)  # sg=1 для Skip-Gram

# Сохранение модели
model.save("skipgram.model")

# Получение частот слов
word_freq = {word: model.wv.get_vecattr(word, "count") for word in model.wv.index_to_key}

# Сортировка слов по частоте
sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

# Функция для визуализации топ-N слов
def visualize_top_n_words(top_n, title):
    # Выбор топ-N слов
    top_words = [word for word, freq in sorted_words[:top_n]]

    # Получение векторов для топ-N слов
    word_vectors = [model.wv[word] for word in top_words]

    # Преобразуем список в массив NumPy
    word_vectors = np.array(word_vectors)

    # Применение TSNE для уменьшения размерности
    tsne = TSNE(n_components=2, random_state=42)
    word_vectors_2d = tsne.fit_transform(word_vectors)

    # Визуализация
    plt.figure(figsize=(10, 10))
    for i, word in enumerate(top_words):
        plt.scatter(word_vectors_2d[i, 0], word_vectors_2d[i, 1])
        plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]))
    plt.title(title)
    plt.show()

# Визуализация для топ-100 слов
visualize_top_n_words(100, "Визуализация эмбеддингов при помощи TSNE (топ-100 слов)")

# Визуализация для топ-500 слов
visualize_top_n_words(500, "Визуализация эмбеддингов при помощи TSNE (топ-500 слов)")

# Визуализация для топ-1000 слов
visualize_top_n_words(1000, "Визуализация эмбеддингов при помощи TSNE (топ-1000 слов)")

# Визуализация для всего словаря
visualize_top_n_words(len(sorted_words), "Визуализация эмбеддингов при помощи TSNE (весь словарь)")
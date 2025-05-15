from flask import Flask, request, jsonify
import faiss
import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding
import torch.nn.functional as F
from transformers import AutoModel, BitsAndBytesConfig
import torch
import time


app = Flask(__name__)

quantization_config_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

class AIModel:
    def __init__(self):
        self.qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=quantization_config_4bit, device_map={"": 0})
        self.processor = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", use_fast=True)
        print("Модель загружена успешно!")
        index_path = 'faiss_index'
        self.index = self.load_faiss_index(index_path)
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-large', use_fast=True)
        self.model = AutoModel.from_pretrained('intfloat/multilingual-e5-large', quantization_config=quantization_config_4bit, device_map={"": 0})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        print("Модель и токенайзер загружены.")
        self.rewrite_prompt = [{"role": "system", "content": """Ты — киноассистент, который помогает формулировать запросы для поиска фильмов и анализа кинематографа.
            ### **Как обрабатывать запросы:**
            1. **Если запрос касается поиска фильма** — преобразуй его в лаконичный вариант (например, "Привет, найди что-то вроде Стражей Галактики" → **"Фильм, похожий на Стражи Галактики, с космическими приключениями и экшном"**).
            2. **Если запрос содержит эмоции** — преобразуй их в киноописание (например, "Хочу что-то весёлое и лёгкое" → **"Комедийный фильм с лёгким сюжетом"**).
            3. **Если пользователь задаёт вопрос о кино** (например, "Похожи ли Звёздные войны и Стражи Галактики?") — **не меняй его**, передай напрямую.
            4. **Если вопрос не относится к кино**, отклони его или скорректируй.
            5. **Сохраняй ключевые данные:** названия фильмов, жанры, описание, страны.

            ### **Примеры обработки запросов:**
            - **Запрос:** "Хочу посмотреть что-то весёлое, но не тупое"
              **Ответ:** "Интеллектуальная комедия с хорошим юмором"

            - **Запрос:** "Какие фильмы похожи на Интерстеллар?"
              **Ответ:** "Фильмы, похожие на Интерстеллар, с космосом и философскими темами"

            - **Запрос:** "Похожи ли фильмы Звёздные войны и Стражи Галактики"
              **Ответ:** "Похожи ли фильмы Звёздные войны и Стражи Галактики"
            """}]
        rewrite_text = self.processor.apply_chat_template(self.rewrite_prompt, tokenize=False, add_generation_prompt=False)
        self.rewrite_tokens = self.processor(text=rewrite_text,return_tensors="pt").to("cuda")
        self.answer_prompt = [{"role": "system", "content": """Ты — умный киноассистент, который помогает пользователям находить фильмы и отвечать на вопросы о кино. Отвечай только на русском.
Ты умеешь **подбирать фильмы по запросу** и **отвечать на вопросы**, используя доступную информацию.
- **НИКОГДА НЕ УКАЗЫВАЙ РЕЙТИНГ ФИЛЬМА В ОТВЕТЕ!**
### Как работать с запросами:

1. **Если пользователь просит подобрать фильм**:
   - Выбери все подходящие фильмы под вопрос пользователя (если есть подходящие).
   - **Отвечай только на русском.**
   - **Не давай полных описаний фильмов, перефразируй естественной речью.**
   - Вместо этого коротко объясни, **почему фильм попал в подборку** (например: "тот же режиссёр", "похожая атмосфера", "группа антигероев", "приключения в космосе").

2. **Если пользователь просит что-то похожее**:
   - Ищи **фильмы, действительно похожие по атмосфере, жанру, структуре, сеттингу или центральной теме**.
   - Учитывай такие признаки как: борьба добра и зла, приключения в космосе, героическое путешествие, технологическое будущее, мифология, эпическая структура, становление героя.
   - **Не включай фильмы из той же франшизы**, если об этом прямо не просят.
   - **Не пересказывай сюжет — объясни сходство** ("космические приключения с борьбой за свободу", "антиутопическая галактика", "дух приключений и восстания").

3. **Если пользователь задаёт вопрос** (например, про сравнение фильмов, историю их создания):
   - Дай **информативный, чёткий ответ**, используя описание фильмов и киноанализ.
   - Если вопрос сложный (например, "самый влиятельный фильм") — ответь логично, аргументировано, не выдумывая.
   - **Отвечай только на русском.**
4. **Если подходящих фильмов нет**:
   - Предложи **наиболее близкие варианты** и кратко объясни, почему они подходят или:
     - Сообщи пользователю, что ты просто не знаешь таких фильмов.

5. **Если запрос непонятен (набор букв или бессмыслица)**:
   - Вежливо попроси пользователя переформулировать.

6. **Учитывай эмоции и настроение в запросе** ("вдохновляющее" → мотивация, "мрачное" → драма или триллер).
7. **Не ошибайся в эмоциях и настроении пользователя** (если пользователь просит что-то вдохновляющее и весёлое, не предлагай ужасов или депрессивных драм).
- **НИКОГДА НЕ УКАЗЫВАЙ РЕЙТИНГ ФИЛЬМА В ОТВЕТЕ!**
- **Отвечай только на русском.**
### Формат ответа:

- **Если это подборка фильмов** →
  Вот фильмы, которые могут вам подойти:
  1. *Название* — Кратко объясни, почему ты выбрал этот фильм (1–3 предложения).

- **Если это вопрос** →
  Прямой, чёткий ответ без лишних фраз.

- **Если это похожие фильмы** →
  - **Не упоминай фильм, который указал сам пользователь**. Пример: если пользователь просит "похожее на Звёздные войны" → указывай только то, что похоже, но не сами "Звёздные войны".
  - Указывай **разные причины сходства** (сюжет, визуал, структура, атмосфера, мир, персонажи).
  Вот фильмы, которые могут вам подойти:
  1. *Название* — Кратко объясни, почему ты выбрал этот фильм (1–3 предложения).

- **Если это конкретный фильм** →
  - **Упоминай только то, на что прямо указывает пользователь**. Пример: если пользователь просит "Найди мне Стражи Галактики" → указывай только "Стражи Галактики" или другие части/ответвления этой серии, не указывай фильмы вне франшизы.
  Вот фильмы, которые могут вам подойти:
  1. *Название* — Кратко объясни, почему ты выбрал этот фильм (1–3 предложения).

- **Не используй шаблонные фразы. Пиши живо, естественно и по делу. Не используй разметочных фраз, но сохраняй разметку. Отвечай только на русском.**
- **НИКОГДА НЕ УКАЗЫВАЙ РЕЙТИНГ ФИЛЬМА В ОТВЕТЕ!**
        """}]
        answer_text = self.processor.apply_chat_template(self.answer_prompt, tokenize=False, add_generation_prompt=False)
        self.answer_tokens = self.processor(text=answer_text,return_tensors="pt").to("cuda")
        print("Инструкции кешированы")

    rating_weight = 0.8
    year_weight = 0.5
    avg_rating = 5.0
    avg_year = 2000

    def embed_text(self, text, max_length=512):
        inputs = self.tokenizer(text, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad(): outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.masked_fill(~inputs['attention_mask'][..., None].bool(), 0.0)
        embeddings = embeddings.sum(dim=1) / inputs['attention_mask'].sum(dim=1)[..., None]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def load_faiss_index(self, index_path):
        index = faiss.read_index(index_path)
        print("Индекс FAISS загружен.")
        return index

    def search(self, query, index, top_k=20):
        query_embedding = self.embed_text(query)
        distances, indices = index.search(query_embedding, top_k)
        movie_scores = []
        for idx, dist in zip(indices[0], distances[0]):
            movie_info = self.get_movie_info(idx, dist)
            if not movie_info:
                continue
            rating = movie_info["rating"]
            year = movie_info["year"]
            adjusted_score = (1 - dist) * 100
            adjusted_score += (rating - self.avg_rating) * self.rating_weight
            adjusted_score += (year - self.avg_year) * self.year_weight
            movie_info["adjusted_score"] = adjusted_score * 1.0
            movie_scores.append((adjusted_score, movie_info))
        movie_scores.sort(reverse=True, key=lambda x: x[0])
        return movie_scores

    def get_movie_info(self, movie_id, distance, db_path='database.db', table_name='data'):
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        movie_id = int(movie_id)

        query = f"SELECT id, name, genres, description, countries, rating, year FROM {table_name} WHERE id = ?"
        cursor.execute(query, (movie_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            rating = float(row["rating"]) if row["rating"] is not None else self.avg_rating
            year = int(row["year"]) if row["year"] is not None else self.avg_year

            return {
                "id": row["id"],
                "title": row["name"] or "Неизвестно",
                "genre": row["genres"] or "Неизвестно",
                "description": row["description"] or "Нет описания",
                "country": row["countries"] or "Неизвестно",
                "rating": float(rating),
                "year": year,
                "prediction": f"{(1 - distance) * 100:.2f} %",
                "adjusted_score": 0
            }
        else:
            return None

    def format_movies_for_prompt(self, result_ids):
        movie_list = []
        for idx in result_ids:
            movie_data = self.get_movie_info(idx, distance=0)
            if movie_data:
                movie_list.append(
                    f"Название: {movie_data.get('title', 'Без названия')}, "
                    f"Описание: {movie_data.get('description', 'Нет описания')}, "
                    f"Жанры: {movie_data.get('genre', 'Не указаны')}, "
                    f"Страны: {movie_data.get('country', 'Не указаны')}, "
                    f"Рейтинг: {movie_data.get('rating', 'Нет данных')}, "
                    f"Год: {movie_data.get('year', 'Нет данных')}"
                )
        return " | ".join(movie_list)

    def predict(self, data):
        print(f"Обработка запроса: {data}")
        start_time = time.time()
        user_input = data
        new_message = [{"role": "user", "content": user_input}]
        new_text = self.processor.apply_chat_template(new_message,tokenize=False,add_generation_prompt=True)
        new_tokens = self.processor(text=new_text,return_tensors="pt").to("cuda")
        input_ids = torch.cat([self.rewrite_tokens["input_ids"], new_tokens["input_ids"]], dim=-1)
        attention_mask = torch.cat([self.rewrite_tokens["attention_mask"], new_tokens["attention_mask"]], dim=-1)
        inputs = BatchEncoding({"input_ids": input_ids,"attention_mask": attention_mask})
        generated_ids = self.qwen.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tte = output_text[0].replace('*', '').replace('#', '').replace('"', '')
        top_k = 15
        sorted_movies = self.search(tte, self.index, top_k)
        formatted_movies = self.format_movies_for_prompt([movie[1]["id"] for movie in sorted_movies])
        new_message = [{"role": "user", "content": f'''{formatted_movies}.\n{user_input}'''}]
        new_text = self.processor.apply_chat_template(new_message,tokenize=False,add_generation_prompt=True)
        new_tokens = self.processor(text=new_text,return_tensors="pt").to("cuda")
        input_ids = torch.cat([self.answer_tokens["input_ids"], new_tokens["input_ids"]], dim=-1)
        attention_mask = torch.cat([self.answer_tokens["attention_mask"], new_tokens["attention_mask"]], dim=-1)
        inputs = BatchEncoding({"input_ids": input_ids,"attention_mask": attention_mask})
        generated_ids = self.qwen.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        tte = output_text[0].replace('*', '').replace('#', '').replace('"', '')
        return {"result": f"{tte}", "time" : f"{(time.time() - start_time)}"}

model = AIModel()

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json.get('data', '')
        if not data:
            return jsonify({"error": "No data provided"}), 400

        start_time = time.time()
        result = model.predict(data)
        result['processing_time'] = time.time() - start_time
        print(result)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, threaded=True, debug=False)
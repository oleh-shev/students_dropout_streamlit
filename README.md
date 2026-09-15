# Student Dropout — Streamlit

Проста форма для навченої моделі з `Students_Dropout_Simple.ipynb`.
10 полів → кнопка «Передбачити» → клас і ймовірності Dropout / Enrolled / Graduate.
Модель уже включена, навчання та CSV для запуску не потрібні.

## Завантаження на GitHub і Streamlit Cloud

1. Створіть порожній репозиторій на GitHub.
2. Через **Add file → Upload files** завантажте вміст цієї папки. `app.py`,
   `requirements.txt` та папка `models` повинні бути в корені репозиторію.
3. Відкрийте [Streamlit Community Cloud](https://share.streamlit.io/) і підключіть GitHub.
4. Натисніть **Create app**, виберіть репозиторій та вашу гілку.
5. У **Main file path** вкажіть `app.py`.
6. В **Advanced settings** виберіть **Python 3.12** і натисніть **Deploy**.

Секрети й API-ключі не потрібні. Після запуску отримаєте посилання на застосунок.
Інструкції платформи: [розгортання](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy),
[залежності](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies).

## Локальний запуск

Потрібен Python 3.12. У терміналі в цій папці:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

На Windows замість `source` використайте `.venv\Scripts\activate`.

## Файли

```text
app.py
requirements.txt
README.md
models/
  students_model.keras
  students_preprocessor.joblib
  students_labels.joblib
```

Файли моделі, перетворень і класів скопійовано разом зі спрощеної лабораторної.
Щоб використати нове навчання, замініть усі три файли одночасно; ознаки повинні
відповідати поточній формі. Версії ML-бібліотек відповідають середовищу навчання.
Використано Keras із PyTorch backend на CPU.

## Значення полів і результат

- Вступний бал: шкала 0–200; семестрові оцінки: 0–20, як у вихідному CSV.
- Course: код освітньої програми з датасету, а не номер року навчання.
- Стать кодується як у CSV: 0 — жіноча, 1 — чоловіча.
- Стипендія й своєчасна оплата: 1 — так, 0 — ні.
- Початкові значення форми повторюють приклад із блокнота: Graduate, приблизно 84,9%.

Модель використовує результати двох семестрів, тому це не прогноз під час вступу.
Тестова accuracy у лабораторній — 74,1%, macro-F1 — 0,6481; найслабше
розпізнається Enrolled. Це навчальна демонстрація.

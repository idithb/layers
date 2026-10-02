# Layers – הפרדת תמונה לשכבות

כלי שמקבל תמונה (פוסטר, מודעה, עטיפה וכו'), מפריד אותה לשכבות – **רקע**, **תמונה/אובייקטים** ו**טקסט** –
מאפשר להזיז ולהגדיל כל שכבה, לשנות את גודל התמונה הסופית (למשל מ-1:1 ל-9:16) ולייצא PNG.

| שלב | כלי | מה קורה |
|---|---|---|
| טקסט | **OpenCV** | איתור קווי מתאר של אותיות (Canny), קיבוץ לשורות ולפסקאות, הפרדת האותיות מהרקע לפי מרחק צבע (כולל "ניקוי" שולי anti-aliasing) |
| תמונה | **BiRefNet** (ONNX) | מסכה רכה של האובייקט הבולט. כל אובייקט נפרד (למשל עטיפת ספר ולוגו) הופך לשכבה משלו |
| רקע | **LaMa** (Big-LaMa) | ציור מחדש של מה שהיה מאחורי הטקסט והאובייקטים, כולל הצל שלהם |
| שינוי גודל | **LaMa** | במצב "הרחבה" – השלמת השוליים החדשים (outpainting) |

## אתר אינטרנט (Hugging Face Spaces – חינם)

GitHub Action מעלה את הקוד אוטומטית ל-Space בכל עדכון. ה-Space משתמש ב-SDK של **Gradio** (חינמי; Docker דורש מנוי), ו-`space_app.py` מריץ בו את השרת הרגיל.
הגדרה חד-פעמית:

1. ב-huggingface.co: **New Space** → שם (למשל `layers`) → SDK: **Gradio** → Blank → Hardware: **CPU basic (free)** → Public → Create.
2. ב-huggingface.co/settings/tokens: **Create new token** → סוג **Write** → העתק אותו.
3. ב-GitHub, בריפו: **Settings → Secrets and variables → Actions**:
   - לשונית **Secrets** → New repository secret → שם `HF_TOKEN`, ערך: הטוקן.
   - לשונית **Variables** → New repository variable → שם `HF_SPACE`, ערך: `שם-המשתמש/layers`.
4. ב-GitHub: **Actions → Deploy to Hugging Face Space → Run workflow**.

אחרי כ-10-15 דקות של בנייה (וכ-2 דקות בהפעלה הראשונה להורדת המודלים) האתר זמין בכתובת `https://huggingface.co/spaces/<שם-המשתמש>/layers`.
מכאן כל push לענף מעדכן את האתר לבד.

הערות: השרת החינמי (2 מעבדים, 16GB) מעבד תמונה בכ-1.5-3 דקות, ומשתמשים מחכים בתור אחד אחרי השני.
Space שלא היה בו שימוש 48 שעות "נרדם" ומתעורר בכניסה הבאה (כדקה).

## התקנה מקומית

```bash
pip install -r requirements.txt
python scripts/download_models.py      # ~1.2GB: BiRefNet (970MB) + Big-LaMa (200MB) לתיקייה models/
```

הקבצים יורדים מ-GitHub Releases:
- `birefnet-general.onnx` – [rembg releases](https://github.com/danielgatis/rembg/releases/tag/v0.0.0)
- `big-lama.pt` – [simple-lama-inpainting releases](https://github.com/enesmsahin/simple-lama-inpainting/releases/tag/v0.1.0)

אפשר לשים את המודלים בתיקייה אחרת עם `LAYERS_MODELS_DIR=/path/to/models`.

## הרצה

```bash
uvicorn app.server:app --port 8000
```

ופתח http://localhost:8000

1. **העלאה** – גרור תמונה ולחץ "הפרד לשכבות".
2. **שכבות** – לחיצה על שכבה בוחרת אותה; גרירה מזיזה; הריבוע בפינה משנה גודל; חיצים מזיזים בפיקסל (Shift = 10).
   👁 מסתיר/מציג, ▲▼ משנים סדר, "מזג לרקע" מצייר שכבה לתוך הרקע (שימושי אם משהו זוהה בטעות), Delete מוחק.
3. **גודל התמונה** – בחר יעד (1:1, 4:5, 9:16, 16:9, באנר או ידני) ואופן התאמה:
   - **הרחבה** – הרקע לא נחתך, LaMa משלים את השטח החדש
   - **מילוי** – הרקע מוגדל וחותכים עודפים
   - **מתיחה** – שינוי יחס
   השכבות זזות יחד עם הרקע, ואחר כך אפשר לסדר אותן מחדש.
4. **ייצוא** – PNG של התוצאה, או כל שכבה כ-PNG שקוף בנפרד.

## ביצועים

- על **CPU**: הפרדה של תמונה 1080×1080 לוקחת כדקה (BiRefNet הוא רוב הזמן). שינוי גודל – 10-15 שניות.
- **זיכרון**: BiRefNet ב-1024×1024 צריך כ-8GB RAM בשיא; הזיכרון משתחרר אחרי כל ריצה. מומלץ 12GB+.
- עם **GPU** (CUDA) זה מהיר בהרבה – התקן `onnxruntime-gpu` ו-torch עם CUDA; הקוד בוחר GPU אוטומטית.
- אם מודל חסר הכלי ממשיך לעבוד: בלי BiRefNet אין שכבת תמונה, בלי LaMa הרקע משוחזר עם OpenCV (איכות נמוכה יותר).

## API

| נקודה | קלט | פלט |
|---|---|---|
| `POST /api/separate` | `image`, `detect_text`, `detect_subject`, `merge_text_lines` | `{job}` – מזהה עבודה |
| `POST /api/resize` | `background`, `width`, `height`, `mode` (`extend`/`cover`/`stretch`) | `{job}` |
| `GET /api/jobs/{job}` | – | `queued`/`running` (+ מקום בתור), `error`, או `done` + תוצאה: רקע + שכבות כ-PNG עם מיקום / רקע חדש + `transform` (`sx, sy, ox, oy`) |
| `GET /api/status` | – | אילו מודלים קיימים |

## מבנה

```
app/text_detect.py   איתור והפרדת טקסט (OpenCV)
app/models.py        BiRefNet + LaMa (טעינה עצלה, fallback)
app/pipeline.py      תמונה → שכבות; שינוי גודל רקע
app/server.py        FastAPI
static/              העורך (HTML/JS, ללא תלויות)
tests/               בדיקות מהירות (המודלים מוחלפים ב-stub)
```

```bash
python -m pytest -q
```

## מגבלות ידועות

- איתור הטקסט מבוסס היוריסטיקות: עובד טוב על טקסט גרפי על רקע אחיד יחסית; על רקע צילומי עמוס יכול לפספס או לזהות בטעות.
  טקסט שנמצא בתוך אובייקט (למשל כותרת על עטיפת ספר) נשאר חלק מהאובייקט.
- LaMa מצוין לרקעים חלקים/מרקמים; לשחזור מבנים מורכבים גדולים (פנים, טקסט שלם) הוא לא מיועד.

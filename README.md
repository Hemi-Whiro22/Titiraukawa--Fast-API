| Route        | Purpose                                            |
| ------------ | -------------------------------------------------- |
| `/status`    | Health / env report for Render pings               |
| `/memory`    | Async insert into Supabase PG                      |
| `/webhook`   | Incoming event hook for Render, Supabase, Firebase |
| `/chat`      | Generic conversation route (gpt-4o-mini)           |
| `/scribe`    | Logs text + runs “Rongo Whisper” background AI     |
| `/speak`     | Async ElevenLabs TTS with stored file              |
| `/translate` | Fast translation endpoint                          |
| `/ocr`       | Google Vision OCR                                  |
| `/static/*`  | Serves TTS + Scribe output                         |

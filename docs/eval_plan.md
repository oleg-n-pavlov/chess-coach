# Eval Plan: оценка качества комментариев

## 1. Сбор датасета

### Источники тренерских комментариев:
1. **Lichess Studies API** — лучший источник
   - `GET https://lichess.org/api/study/{studyId}.pgn?comments=true&variations=true`
   - Fischer "My 60 Memorable Games": study ID `nk2t0m1n`
   - Искать популярные studies по топикам
2. **BeginChess.com** — PGN из классических книг
   - Chernev "Logical Chess Move by Move": https://beginchess.com/games/lcmbm.pgn
   - Nimzowitsch "My System": https://s3.amazonaws.com/beginchess/pgns/mysystem_pgn.zip
   - Shereshevsky "Endgame Strategy": https://s3.amazonaws.com/beginchess/pgns/shereshevsky_endgame_strategy.pgn
   - Informant 100 Golden Games: https://s3.amazonaws.com/beginchess/pgns/informant_100goldengames.pgn
3. **GitHub: ValdemarOrn/Chess** — аннотированные партии GM
   - https://github.com/ValdemarOrn/Chess/blob/master/Annotated%20Games/GM_games.pgn
4. **Jhamtani dataset** — 298K пар ход-комментарий с GameKnot
   - https://github.com/harsh19/ChessCommentaryGeneration

### Формат PGN-комментариев:
```pgn
8. Bh6 {Kasparov exchanges the dark-squared bishops,
a typical strategy in the Pirc complex.} 8... Bxh6
10. a3 {!} {A quiet but deep move. White prevents ...b4}
```
- `{text}` — текстовые комментарии
- `(moves)` — альтернативные варианты
- NAG: `$1` = !, `$2` = ?, `$4` = !!, `$6` = ?!

## 2. Парсинг и конвертация

Скрипт `src/eval/parse_annotated_pgn.py`:
- Читает annotated PGN через python-chess
- Для каждого хода с комментарием извлекает:
  ```json
  {
    "fen": "...",
    "move_san": "Bh6",
    "move_number": 8,
    "side": "white",
    "human_comment": "Kasparov exchanges the dark-squared bishops...",
    "nag": ["$1"],
    "variations": ["10...b4 11.Nc3"]
  }
  ```
- Сохраняет в `data/eval/human_annotations.jsonl`

## 3. Прогон нашего пайплайна

Скрипт `src/eval/run_pipeline.py`:
- Для каждой позиции из human_annotations.jsonl:
  - Прогнать через PositionAnalyzer (SF + Leela concepts)
  - Сгенерировать комментарий через CommentaryGenerator (Haiku)
  - Сохранить в `data/eval/our_annotations.jsonl`:
    ```json
    {
      "fen": "...",
      "move_san": "Bh6",
      "human_comment": "...",
      "our_comment": "...",
      "eval_loss": 5,
      "move_quality": "excellent",
      "key_concepts": [...],
      "concept_diff": {...}
    }
    ```

## 4. Метрики сравнения

Скрипт `src/eval/compare.py`:

### Автоматические метрики:
- **Тематическое покрытие**: упомянули ли мы те же концепты что тренер
  (king safety, tactics, pawn structure и т.п.)
- **Фактическая корректность**: не противоречит ли наш комментарий данным движка
- **Согласованность оценки**: тренер говорит "ошибка" — мы тоже?
- **BLEU/ROUGE** (для справки, не основная метрика)

### Ручная оценка (выборка 50-100 позиций):
- Полезность комментария (1-5)
- Корректность (0/1)
- Информативность vs тренера (хуже/равно/лучше)

## 5. Итеративное улучшение

По результатам eval:
- Тюнить системный промпт (добавить примеры, уточнить стиль)
- Увеличить training data для SVM (сейчас 500 позиций → 5000+)
- Добавить недостающие концепты (material, imbalance, pawns — 7 из 24 не обучены)
- Сравнить Haiku vs Sonnet на eval dataset

"""Generate chess commentary using Claude, guided by concept vectors.

Takes structured analysis (Stockfish eval + concept diff) and produces
natural language commentary in the style of a Soviet chess trainer.
"""

import anthropic


SYSTEM_PROMPTS = {
    "ru": """Ты — опытный шахматный тренер советской школы. Твоя задача — комментировать ходы партии на русском языке.

ПРАВИЛА:
1. Комментируй ТОЛЬКО на основе предоставленных данных. НЕ выдумывай ходы, фигуры или позиции.
2. Объясняй ПОЧЕМУ ход хороший или плохой, используя концепты (безопасность короля, мобильность, пешечная структура и т.д.)
3. Если ход — ошибка, объясни что было лучше и почему.
4. Используй шахматную терминологию: "инициатива", "компенсация", "давление", "слабость", "активность фигур".
5. Длина комментария: 1-3 предложения. Не больше.
6. Не используй eval числа напрямую. Переводи их в понятные оценки: "небольшое преимущество", "решающий перевес", "равенство".
7. Если ход лучший по движку — достаточно короткой заметки о позиции.
8. Используй markdown форматирование: **жирный** для ключевых ходов (например **1.e4**) и важных позиционных терминов.

КОНЦЕПТЫ И ИХ ЗНАЧЕНИЕ:
- king_safety: безопасность короля (высокое = король защищён, низкое = король под атакой)
- mobility: подвижность фигур (высокое = фигуры активны)
- space: контроль пространства (преимущество в центре и на фланге)
- threats: угрозы (незащищённые фигуры, тактические мотивы)
- passed_pawns: проходные пешки (близость к превращению)
- bishops/knights/rooks/queens: активность конкретных фигур
- pawns: пешечная структура (изолированные, сдвоенные пешки)
- material: материальное соотношение""",

    "en": """You are an experienced chess coach. Comment on moves in English.

RULES:
1. Comment ONLY based on the provided data. DO NOT invent moves, pieces, or positions.
2. Explain WHY a move is good or bad using concepts (king safety, mobility, pawn structure, etc.)
3. If the move is a mistake, explain what was better and why.
4. Use chess terminology: "initiative", "compensation", "pressure", "weakness", "piece activity".
5. Commentary length: 1-3 sentences. No more.
6. Don't use eval numbers directly. Translate them: "slight advantage", "decisive edge", "equality".
7. If the move is best by engine — a short positional note is enough.
8. Use markdown formatting: **bold** for key moves (e.g. **1.e4**) and important positional terms.

CONCEPTS AND THEIR MEANING:
- king_safety: king safety (high = king is safe, low = king under attack)
- mobility: piece mobility (high = pieces are active)
- space: space control (center and flank advantage)
- threats: threats (unprotected pieces, tactical motifs)
- passed_pawns: passed pawns (proximity to promotion)
- bishops/knights/rooks/queens: activity of specific pieces
- pawns: pawn structure (isolated, doubled pawns)
- material: material balance""",
}

# Keep backward compat
SYSTEM_PROMPT = SYSTEM_PROMPTS["ru"]

MOVE_PROMPT_TEMPLATES = {
    "ru": """Позиция (FEN): {fen}
Номер хода: {move_number}
Сторона: {side}
Сыгранный ход: {played_move}
Оценка после хода: {played_eval}
Лучший ход: {best_move} (оценка: {best_eval})
Качество хода: {move_quality}
Потеря в оценке: {eval_loss} сотых пешки
{extra_info}
Главные изменения в позиции после хода:
{concept_changes}

Прокомментируй этот ход.""",

    "en": """Position (FEN): {fen}
Move number: {move_number}
Side: {side}
Played move: {played_move}
Evaluation after move: {played_eval}
Best move: {best_move} (eval: {best_eval})
Move quality: {move_quality}
Eval loss: {eval_loss} centipawns
{extra_info}
Key position changes after the move:
{concept_changes}

Comment on this move.""",
}

MOVE_PROMPT_TEMPLATE = MOVE_PROMPT_TEMPLATES["ru"]

SIDE_NAMES = {"ru": {"w": "белые", "b": "чёрные"}, "en": {"w": "white", "b": "black"}}

EVAL_LABELS = {
    "ru": [
        (300, "решающее преимущество белых"),
        (150, "большое преимущество белых"),
        (50, "преимущество белых"),
        (-50, "примерное равенство"),
        (-150, "преимущество чёрных"),
        (-300, "большое преимущество чёрных"),
        (None, "решающее преимущество чёрных"),
    ],
    "en": [
        (300, "decisive white advantage"),
        (150, "large white advantage"),
        (50, "white advantage"),
        (-50, "roughly equal"),
        (-150, "black advantage"),
        (-300, "large black advantage"),
        (None, "decisive black advantage"),
    ],
}


def format_eval(cp: int | None, lang: str = "ru") -> str:
    """Format centipawn evaluation to human-readable string."""
    if cp is None:
        return "неизвестно" if lang == "ru" else "unknown"
    labels = EVAL_LABELS.get(lang, EVAL_LABELS["ru"])
    for threshold, label in labels:
        if threshold is None or cp > threshold:
            return label
    return labels[-1][1]


CONCEPT_NAMES_RU = {
    "king_safety_white": "безопасность белого короля",
    "king_safety_black": "безопасность чёрного короля",
    "mobility_white": "мобильность белых",
    "mobility_black": "мобильность чёрных",
    "space_white": "пространство белых",
    "space_black": "пространство чёрных",
    "threats_white": "угрозы белых",
    "threats_black": "угрозы чёрных",
    "passed_pawns_white": "проходные пешки белых",
    "passed_pawns_black": "проходные пешки чёрных",
    "bishops_white": "активность белых слонов",
    "bishops_black": "активность чёрных слонов",
    "knights_white": "активность белых коней",
    "knights_black": "активность чёрных коней",
    "rooks_white": "активность белых ладей",
    "rooks_black": "активность чёрных ладей",
    "queens_white": "активность белого ферзя",
    "queens_black": "активность чёрного ферзя",
    "pawns_white": "пешечная структура белых",
    "pawns_black": "пешечная структура чёрных",
    "material_white": "материал белых",
    "material_black": "материал чёрных",
    "imbalance_white": "дисбаланс (белые)",
    "imbalance_black": "дисбаланс (чёрные)",
}


def format_concept_changes(key_concepts: list[dict]) -> str:
    """Format concept changes for the prompt."""
    if not key_concepts:
        return "Нет значимых изменений."

    lines = []
    for kc in key_concepts:
        concept = kc["concept"]
        change = kc["change"]
        name_ru = CONCEPT_NAMES_RU.get(concept, concept)
        direction = "улучшилась" if change > 0 else "ухудшилась"
        magnitude = abs(change)
        if magnitude > 2:
            strength = "значительно"
        elif magnitude > 1:
            strength = "заметно"
        else:
            strength = "немного"
        lines.append(f"- {name_ru}: {strength} {direction} ({change:+.2f})")

    return "\n".join(lines)


class CommentaryGenerator:
    """Generate chess commentary using Claude API."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def comment_move(
        self, analysis: dict, move_number: int, lang: str = "ru", force: bool = False,
    ) -> str:
        """Generate a commentary for a single move.

        Args:
            analysis: Output from PositionAnalyzer.analyze()
            move_number: Move number in the game.
            lang: Language code ("ru" or "en").
            force: If True, always generate (don't skip best moves).

        Returns:
            Commentary string.
        """
        if "played_move" not in analysis:
            return ""

        # Skip best moves (don't over-comment) — unless forced
        if not force and analysis.get("is_best") and analysis.get("eval_loss", 0) <= 5:
            return ""

        side_key = "w" if "w" in analysis["fen"].split()[1] else "b"
        side = SIDE_NAMES.get(lang, SIDE_NAMES["ru"])[side_key]
        concept_text = format_concept_changes(analysis.get("key_concepts", []))

        best_move_info = analysis.get("best_moves", [{}])
        best_san = best_move_info[0]["san"] if best_move_info else "?"
        best_eval_cp = best_move_info[0]["eval"] if best_move_info else 0

        # Build extra context lines
        extra_lines = []

        # Opening info
        opening = analysis.get("opening")
        if opening and opening.get("name"):
            total = opening.get("total_games", 0)
            w = opening.get("white_wins", 0)
            d = opening.get("draws", 0)
            b = opening.get("black_wins", 0)
            if lang == "en":
                extra_lines.append(
                    f"Opening: {opening['name']} (ECO {opening.get('eco', '?')}). "
                    f"Master stats: +{w}={d}-{b} ({total} games)"
                )
            else:
                extra_lines.append(
                    f"Дебют: {opening['name']} (ECO {opening.get('eco', '?')}). "
                    f"Статистика мастеров: +{w}={d}-{b} ({total} партий)"
                )

        # Mate info
        mate_in = analysis.get("played_mate_in")
        if mate_in is not None:
            if lang == "en":
                if mate_in > 0:
                    extra_lines.append(f"White mates in {mate_in}")
                elif mate_in < 0:
                    extra_lines.append(f"Black mates in {-mate_in}")
            else:
                if mate_in > 0:
                    extra_lines.append(f"Мат белыми в {mate_in} ходов")
                elif mate_in < 0:
                    extra_lines.append(f"Мат чёрными в {-mate_in} ходов")

        # Tablebase
        tb = analysis.get("tablebase")
        if tb:
            if lang == "en":
                extra_lines.append(
                    f"Tablebase: {tb['result']} (DTZ: {tb.get('dtz', '?')})"
                )
            else:
                tb_result = {"win": "выигрыш", "draw": "ничья", "loss": "проигрыш"}
                extra_lines.append(
                    f"Таблица эндшпилей: {tb_result.get(tb['result'], tb['result'])} "
                    f"(DTZ: {tb.get('dtz', '?')})"
                )

        extra_info = "\n".join(extra_lines)
        if extra_info:
            extra_info = "\n" + extra_info

        template = MOVE_PROMPT_TEMPLATES.get(lang, MOVE_PROMPT_TEMPLATES["ru"])
        prompt = template.format(
            fen=analysis["fen"],
            move_number=move_number,
            side=side,
            played_move=analysis["played_move"],
            played_eval=format_eval(analysis.get("played_eval"), lang),
            best_move=best_san,
            best_eval=format_eval(best_eval_cp, lang),
            move_quality=analysis["move_quality"],
            eval_loss=analysis.get("eval_loss", 0),
            extra_info=extra_info,
            concept_changes=concept_text,
        )

        system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ru"])
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()

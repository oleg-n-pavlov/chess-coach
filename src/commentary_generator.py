"""Generate chess commentary using Claude, guided by concept vectors.

Takes structured analysis (Stockfish eval + concept diff) and produces
natural language commentary in the style of a Soviet chess trainer.
"""

import re

import anthropic


# ─── Piece name mappings for human-readable commentary ───────────

PIECE_NAMES = {
    "ru": {"K": "король", "Q": "ферзь", "R": "ладья", "B": "слон", "N": "конь"},
    "en": {"K": "king", "Q": "queen", "R": "rook", "B": "bishop", "N": "knight"},
}

CAPTURE_WORD = {"ru": "бьёт на", "en": "captures on"}
MOVES_TO_WORD = {"ru": "идёт на", "en": "goes to"}
PAWN_NAME = {"ru": "пешка", "en": "pawn"}
CASTLES_SHORT = {"ru": "короткая рокировка", "en": "kingside castling"}
CASTLES_LONG = {"ru": "длинная рокировка", "en": "queenside castling"}


def san_to_human(san: str, lang: str = "ru") -> str:
    """Convert SAN notation to human-readable piece description.

    Examples (ru):
      "Nf3"  -> "конь идёт на f3"
      "Bxe5" -> "слон бьёт на e5"
      "e4"   -> "пешка идёт на e4"
      "O-O"  -> "короткая рокировка"
      "Qxd7+"-> "ферзь бьёт на d7"
    """
    # Strip check/mate markers and annotations
    clean = san.rstrip("+#!?")

    # Castling
    if clean in ("O-O", "0-0"):
        return CASTLES_SHORT.get(lang, CASTLES_SHORT["en"])
    if clean in ("O-O-O", "0-0-0"):
        return CASTLES_LONG.get(lang, CASTLES_LONG["en"])

    pieces = PIECE_NAMES.get(lang, PIECE_NAMES["en"])
    capture = CAPTURE_WORD.get(lang, CAPTURE_WORD["en"])
    moves_to = MOVES_TO_WORD.get(lang, MOVES_TO_WORD["en"])
    pawn = PAWN_NAME.get(lang, PAWN_NAME["en"])

    # Parse SAN: optional piece letter, optional disambiguation, optional 'x', destination
    m = re.match(r'^([KQRBN])?([a-h]?[1-8]?)?(x)?([a-h][1-8])(=[QRBN])?', clean)
    if not m:
        return san  # fallback to original

    piece_char = m.group(1)
    _disambig = m.group(2)
    is_capture = m.group(3) is not None
    dest = m.group(4)
    promotion = m.group(5)

    piece_name = pieces.get(piece_char, pawn) if piece_char else pawn
    action = capture if is_capture else moves_to

    result = f"{piece_name} {action} {dest}"

    if promotion:
        promo_piece = pieces.get(promotion[1:], promotion[1:])
        if lang == "ru":
            result += f" с превращением в {promo_piece}"
        else:
            result += f" promoting to {promo_piece}"

    return result


def san_list_to_human(san_list: list[str], lang: str = "ru") -> str:
    """Convert a list of SAN moves to human-readable line.

    E.g., ["e4", "e5", "Nf3"] -> "пешка e4, пешка e5, конь f3"
    Uses abbreviated form (just piece + square) for PV lines.
    """
    pieces = PIECE_NAMES.get(lang, PIECE_NAMES["en"])
    pawn = PAWN_NAME.get(lang, PAWN_NAME["en"])
    parts = []
    for san in san_list:
        clean = san.rstrip("+#!?")
        if clean in ("O-O", "0-0"):
            parts.append(CASTLES_SHORT.get(lang, CASTLES_SHORT["en"]))
            continue
        if clean in ("O-O-O", "0-0-0"):
            parts.append(CASTLES_LONG.get(lang, CASTLES_LONG["en"]))
            continue
        m = re.match(r'^([KQRBN])?([a-h]?[1-8]?)?(x)?([a-h][1-8])', clean)
        if m:
            piece_char = m.group(1)
            dest = m.group(4)
            is_capture = m.group(3) is not None
            piece_name = pieces.get(piece_char, pawn) if piece_char else pawn
            cap = "x" if is_capture else ""
            parts.append(f"{piece_name} {cap}{dest}")
        else:
            parts.append(san)
    return ", ".join(parts)


# ─── Tactical motif formatting ────────────────────────────────────

TACTIC_NAMES = {
    "ru": {
        "fork": "вилка",
        "pin": "связка",
        "skewer": "сквозной удар",
        "discovered_attack": "вскрытое нападение",
        "discovered_check": "вскрытый шах",
    },
    "en": {
        "fork": "fork",
        "pin": "pin",
        "skewer": "skewer",
        "discovered_attack": "discovered attack",
        "discovered_check": "discovered check",
    },
}

PIECE_NAMES_TACTIC = {
    "ru": {
        "pawn": "пешка", "knight": "конь", "bishop": "слон",
        "rook": "ладья", "queen": "ферзь", "king": "король",
    },
    "en": {
        "pawn": "pawn", "knight": "knight", "bishop": "bishop",
        "rook": "rook", "queen": "queen", "king": "king",
    },
}


def _translate_piece_ref(piece_str: str, lang: str) -> str:
    """Translate 'knight d3' -> 'конь d3'."""
    names = PIECE_NAMES_TACTIC.get(lang, PIECE_NAMES_TACTIC["en"])
    for en_name, loc_name in names.items():
        if piece_str.startswith(en_name):
            return piece_str.replace(en_name, loc_name, 1)
    return piece_str


def format_tactics(
    tactics_best: list[dict],
    tactics_played: list[dict],
    tactics_current: list[dict],
    lang: str = "ru",
) -> str:
    """Format detected tactical motifs for the commentary prompt.

    Returns human-readable description of tactics found in:
    - Current position (existing pins, forks, etc.)
    - Best move's PV line (what the best move leads to)
    - Played move's PV line (what the played move leads to)
    """
    tnames = TACTIC_NAMES.get(lang, TACTIC_NAMES["en"])
    lines = []

    def _describe_tactic(t: dict) -> str:
        ttype = t["type"]
        name = tnames.get(ttype, ttype)
        ply = t.get("ply", 0)

        if ttype == "fork":
            attacker = _translate_piece_ref(t["attacker"], lang)
            targets = [_translate_piece_ref(tgt["piece"], lang) for tgt in t.get("targets", [])]
            targets_str = " и ".join(targets) if lang == "ru" else " and ".join(targets)
            if ply > 0:
                moves_word = "ход" if lang == "ru" else "move"
                if lang == "ru":
                    ply_str = f"через {ply} полуход{'а' if ply < 5 else 'ов'}"
                else:
                    ply_str = f"in {ply} half-move{'s' if ply > 1 else ''}"
                return f"**{name}**: {ply_str} {attacker} атакует {targets_str}" if lang == "ru" else \
                       f"**{name}**: {ply_str} {attacker} attacks {targets_str}"
            else:
                return f"**{name}**: {attacker} атакует {targets_str}" if lang == "ru" else \
                       f"**{name}**: {attacker} attacks {targets_str}"

        elif ttype == "pin":
            pinner = _translate_piece_ref(t["pinner"], lang)
            pinned = _translate_piece_ref(t["pinned"], lang)
            behind = _translate_piece_ref(t["behind"], lang)
            if lang == "ru":
                return f"**{name}**: {pinner} связывает {pinned} с {behind}"
            else:
                return f"**{name}**: {pinner} pins {pinned} against {behind}"

        elif ttype == "skewer":
            attacker = _translate_piece_ref(t["attacker"], lang)
            front = _translate_piece_ref(t["front"], lang)
            behind = _translate_piece_ref(t["behind"], lang)
            if lang == "ru":
                return f"**{name}**: {attacker} атакует {front}, за которым стоит {behind}"
            else:
                return f"**{name}**: {attacker} attacks {front} with {behind} behind"

        elif ttype in ("discovered_attack", "discovered_check"):
            revealer = _translate_piece_ref(t["revealer"], lang)
            target = _translate_piece_ref(t["target"], lang)
            if lang == "ru":
                return f"**{name}**: {revealer} атакует {target}"
            else:
                return f"**{name}**: {revealer} attacks {target}"

        return f"**{name}**"

    # Current position tactics
    if tactics_current:
        if lang == "ru":
            lines.append("Тактика на доске:")
        else:
            lines.append("Tactics on the board:")
        for t in tactics_current:
            lines.append(f"  - {_describe_tactic(t)}")

    # Best line tactics
    if tactics_best:
        if lang == "ru":
            lines.append("Тактика в линии лучшего хода:")
        else:
            lines.append("Tactics in the best move line:")
        for t in tactics_best:
            lines.append(f"  - {_describe_tactic(t)}")

    # Played line tactics
    if tactics_played:
        if lang == "ru":
            lines.append("Тактика в линии сыгранного хода:")
        else:
            lines.append("Tactics in the played move line:")
        for t in tactics_played:
            lines.append(f"  - {_describe_tactic(t)}")

    return "\n".join(lines) if lines else ""


# ─── System prompts ──────────────────────────────────────────────

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
8. Используй markdown форматирование: **жирный** для ключевых ходов и важных позиционных терминов.
9. Называй фигуры по-русски: "слон идёт на e3", "конь прыгает на c4", "ферзь бьёт на a4". НЕ используй символы фигур (N, B, R, Q) и алгебраическую нотацию в тексте комментария.
10. НИКОГДА не выдумывай ходы. Упоминай ТОЛЬКО ходы из предоставленных данных: сыгранный ход, лучший ход и линию движка. Если не уверен — не упоминай конкретный ход.
11. Если предоставлена тактическая информация (вилка, связка, сквозной удар, вскрытое нападение) — обязательно упомяни её в комментарии. Это ключевая причина почему ход хороший или плохой. Опиши конкретно: какая фигура атакует, какие фигуры под ударом.

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
8. Use markdown formatting: **bold** for key moves and important positional terms.
9. Use descriptive piece names: "bishop goes to e3", "knight jumps to c4", "queen captures on a4". Do NOT use algebraic symbols (N, B, R, Q) in commentary text.
10. NEVER invent moves. ONLY mention moves from the provided data: the played move, the best move, and engine lines. If unsure — do not mention a specific move.
11. If tactical information is provided (fork, pin, skewer, discovered attack) — you MUST mention it in the commentary. This is the key reason why a move is good or bad. Describe specifically: which piece attacks, which pieces are under threat.

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
    "ru": """Номер хода: {move_number}
Сторона: {side}
Сыгранный ход: {played_move}
Оценка после хода: {played_eval}
Лучший ход: {best_move} (оценка: {best_eval})
Линия движка: {engine_line}
Качество хода: {move_quality}
Потеря в оценке: {eval_loss} сотых пешки
{extra_info}
Главные изменения в позиции после хода:
{concept_changes}
{tactics}
Прокомментируй этот ход.""",

    "en": """Move number: {move_number}
Side: {side}
Played move: {played_move}
Evaluation after move: {played_eval}
Best move: {best_move} (eval: {best_eval})
Engine line: {engine_line}
Move quality: {move_quality}
Eval loss: {eval_loss} centipawns
{extra_info}
Key position changes after the move:
{concept_changes}
{tactics}
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


CONCEPT_NAMES = {
    "ru": {
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
    },
    "en": {
        "king_safety_white": "white king safety",
        "king_safety_black": "black king safety",
        "mobility_white": "white mobility",
        "mobility_black": "black mobility",
        "space_white": "white space",
        "space_black": "black space",
        "threats_white": "white threats",
        "threats_black": "black threats",
        "passed_pawns_white": "white passed pawns",
        "passed_pawns_black": "black passed pawns",
        "bishops_white": "white bishops",
        "bishops_black": "black bishops",
        "knights_white": "white knights",
        "knights_black": "black knights",
        "rooks_white": "white rooks",
        "rooks_black": "black rooks",
        "queens_white": "white queen",
        "queens_black": "black queen",
        "pawns_white": "white pawn structure",
        "pawns_black": "black pawn structure",
        "material_white": "white material",
        "material_black": "black material",
        "imbalance_white": "imbalance (white)",
        "imbalance_black": "imbalance (black)",
    },
}

# Keep backward compat
CONCEPT_NAMES_RU = CONCEPT_NAMES["ru"]

DIRECTION_WORDS = {
    "ru": {"improved": "улучшилась", "worsened": "ухудшилась"},
    "en": {"improved": "improved", "worsened": "worsened"},
}
STRENGTH_WORDS = {
    "ru": {3: "значительно", 2: "заметно", 1: "немного"},
    "en": {3: "significantly", 2: "noticeably", 1: "slightly"},
}


def format_concept_changes(key_concepts: list[dict], lang: str = "ru") -> str:
    """Format concept changes for the prompt."""
    no_changes = "Нет значимых изменений." if lang == "ru" else "No significant changes."
    if not key_concepts:
        return no_changes

    names = CONCEPT_NAMES.get(lang, CONCEPT_NAMES["ru"])
    dirs = DIRECTION_WORDS.get(lang, DIRECTION_WORDS["en"])
    strengths = STRENGTH_WORDS.get(lang, STRENGTH_WORDS["en"])

    lines = []
    for kc in key_concepts:
        concept = kc["concept"]
        change = kc["change"]
        name = names.get(concept, concept)
        direction = dirs["improved"] if change > 0 else dirs["worsened"]
        magnitude = abs(change)
        if magnitude > 2:
            strength = strengths[3]
        elif magnitude > 1:
            strength = strengths[2]
        else:
            strength = strengths[1]
        lines.append(f"- {name}: {strength} {direction} ({change:+.2f})")

    return "\n".join(lines)


class CommentaryGenerator:
    """Generate chess commentary using Claude API."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def comment_move(
        self, analysis: dict, move_number: int, lang: str = "ru", force: bool = False,
    ) -> dict:
        """Generate a commentary for a single move.

        Args:
            analysis: Output from PositionAnalyzer.analyze()
            move_number: Move number in the game.
            lang: Language code ("ru" or "en").
            force: If True, always generate (don't skip best moves).

        Returns:
            Dict with "comment" and "prompt" keys.
        """
        if "played_move" not in analysis:
            return {"comment": "", "prompt": ""}

        # Skip best moves (don't over-comment) — unless forced
        if not force and analysis.get("is_best") and analysis.get("eval_loss", 0) <= 5:
            return {"comment": "", "prompt": ""}

        side_key = "w" if "w" in analysis.get("fen", "w").split()[1] else "b"
        side = SIDE_NAMES.get(lang, SIDE_NAMES["ru"])[side_key]
        concept_text = format_concept_changes(analysis.get("key_concepts", []), lang)

        best_move_info = analysis.get("best_moves", [{}])
        best_san = best_move_info[0]["san"] if best_move_info else "?"
        best_eval_cp = best_move_info[0]["eval"] if best_move_info else 0

        # Convert moves to human-readable
        played_human = san_to_human(analysis["played_move"], lang)
        best_human = san_to_human(best_san, lang)

        # Build engine PV line (human-readable)
        engine_line = ""
        if best_move_info and best_move_info[0].get("pv"):
            pv = best_move_info[0]["pv"][:5]
            engine_line = san_list_to_human(pv, lang)

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

        # Format tactical motifs
        tactics_text = format_tactics(
            analysis.get("tactics_in_best_line", []),
            analysis.get("tactics_in_played_line", []),
            analysis.get("tactics_current", []),
            lang,
        )

        template = MOVE_PROMPT_TEMPLATES.get(lang, MOVE_PROMPT_TEMPLATES["ru"])
        prompt = template.format(
            move_number=move_number,
            side=side,
            played_move=played_human,
            played_eval=format_eval(analysis.get("played_eval"), lang),
            best_move=best_human,
            best_eval=format_eval(best_eval_cp, lang),
            engine_line=engine_line or ("нет данных" if lang == "ru" else "no data"),
            move_quality=analysis["move_quality"],
            eval_loss=analysis.get("eval_loss", 0),
            extra_info=extra_info,
            concept_changes=concept_text,
            tactics=tactics_text,
        )

        system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["ru"])
        response = self.client.messages.create(
            model=self.model,
            max_tokens=300,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "comment": response.content[0].text.strip(),
            "prompt": prompt,
        }

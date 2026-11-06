#!/usr/bin/env bash
# UCI engine wrapper for lichess-bot: maia3 fine-tuned on TsukinoLuna/Sasayaku rapid games.
# lichess-bot launches this as a plain UCI engine; extra args from config are appended.
exec /mnt/General_Data/Projects/Python/maia3/.venv/bin/maia3-uci \
  --model maia3-79m \
  --checkpoint-path /mnt/General_Data/Projects/Python/maia3/finetunes/maia3-79m-v5-final/best.pt \
  --use-uci-history \
  --elo 1750 \
  --temperature 0.4 \
  --opening-book /mnt/General_Data/Projects/Python/maia3/data/books/opening_book_compiled.json \
  --book-depth 12 \
  "$@"

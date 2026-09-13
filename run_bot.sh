#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/soi/kokoro_bot_env/bin/activate
cd "$DIR"

echo "Đang khởi động Telegram Bot bằng môi trường ảo kokoro_bot_env..."
exec python3 bot.py "$@"

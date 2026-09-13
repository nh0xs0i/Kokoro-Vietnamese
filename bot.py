#!/usr/bin/env /home/soi/kokoro_bot_env/bin/python3
"""
Telegram Bot for Kokoro Vietnamese TTS
Tương thích Ubuntu (CPU/GPU) - Tối ưu cho máy CPU yếu, chạy ổn định, không chặn event loop.

Chỉ cần điền TELEGRAM_BOT_TOKEN là có thể chạy ngay!
"""

import asyncio
import io
import logging
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from kokoro_vietnamese import KokoroVietnamese, VOICES, list_voices

# ---------------------------------------------------------
# CẤU HÌNH (CONFIGURATION)
# ---------------------------------------------------------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8389471597:AAGn82h8NC-vEkCBG7uYK3hHFP7-vVAuhxc")

# Thư mục gốc chứa model và voicepacks
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_voices"

# Cấu hình thiết bị: CPU tối ưu cho server/VPS không có GPU
DEVICE = "cpu"

# Giọng đọc mặc định
DEFAULT_VOICE = "diem_trinh"

# Giới hạn độ dài ký tự cho mỗi yêu cầu để tránh quá tải CPU
MAX_TEXT_LENGTH = 500

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)
logger = logging.getLogger("KokoroBot")

# Khóa luồng (lock) đảm bảo mỗi thời điểm chỉ xử lý 1 tác vụ TTS
# Giúp máy CPU yếu không bị treo/quá tải do nhiều request đồng thời
tts_lock = asyncio.Lock()

# Cache model instance theo giọng đọc để không phải tải lại model
_tts_instances: dict[str, KokoroVietnamese] = {}
# Lưu voice đã chọn của từng người dùng {user_id: voice_name}
user_voices: dict[int, str] = {}


def get_tts_engine(voice: str = DEFAULT_VOICE) -> KokoroVietnamese:
    """Khởi tạo hoặc lấy engine KokoroTTS đã cache."""
    if voice not in VOICES:
        voice = DEFAULT_VOICE

    if voice not in _tts_instances:
        logger.info(f"Đang nạp mô hình Kokoro TTS (voice: {voice}, device: {DEVICE})...")
        model_path = BASE_DIR / "kokoro_vi.pth"
        config_path = BASE_DIR / "config.json"
        voicepack_path = BASE_DIR / "voicepacks" / f"{voice}.pt"

        kwargs = {"device": DEVICE, "voice": voice}
        if model_path.exists():
            kwargs["model_path"] = str(model_path)
        if config_path.exists():
            kwargs["config_path"] = str(config_path)
        if voicepack_path.exists():
            kwargs["voicepack_path"] = str(voicepack_path)

        _tts_instances[voice] = KokoroVietnamese(**kwargs)
        logger.info(f"Nạp mô hình thành công cho voice '{voice}'!")

    return _tts_instances[voice]


def _synthesize_sync(text: str, voice: str) -> tuple[bytes, str, float]:
    """Hàm xử lý TTS đồng bộ (chạy trên thread pool để không block asyncio loop)."""
    t_start = time.time()
    engine = get_tts_engine(voice)
    audio, phonemes = engine.synthesize(text, normalize_peak=0.95)

    buffer = io.BytesIO()
    sf.write(buffer, audio, 24000, format="WAV")
    buffer.seek(0)
    elapsed = time.time() - t_start

    return buffer.getvalue(), phonemes, elapsed


# ---------------------------------------------------------
# TELEGRAM HANDLERS
# ---------------------------------------------------------
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /start - Hướng dẫn sử dụng."""
    user = update.effective_user
    current_voice = user_voices.get(user.id, DEFAULT_VOICE)
    voice_label = VOICES.get(current_voice, {}).get("label", current_voice)

    msg = (
        f"👋 Xin chào <b>{user.first_name}</b>!\n\n"
        f"Tôi là Bot chuyển văn bản tiếng Việt thành giọng nói tự nhiên sử dụng <b>Kokoro-Vietnamese</b>.\n\n"
        f"🎙 Giọng hiện tại của bạn: <b>{voice_label}</b> (<code>{current_voice}</code>)\n\n"
        f"<b>Cách dùng:</b>\n"
        f"• Gửi câu/đoạn văn bản tiếng Việt bất kỳ (tối đa {MAX_TEXT_LENGTH} ký tự)\n"
        f"• Gõ <code>/voices</code> để nghe thử mẫu và chọn giọng đọc\n"
        f"• Gõ <code>/voice [mã_giọng]</code> để đổi nhanh giọng đọc\n"
        f"• Gõ <code>/help</code> để xem hướng dẫn chi tiết"
    )
    await update.message.reply_html(msg)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /help."""
    msg = (
        "📖 <b>HƯỚNG DẪN SỬ DỤNG:</b>\n\n"
        "1. <b>Tạo giọng đọc:</b> Chỉ cần gửi trực tiếp câu/đoạn văn cần đọc vào chat.\n"
        "2. <b>Xem và nghe thử giọng mẫu:</b> Gõ <code>/voices</code>. Bot sẽ gửi kèm file âm thanh mẫu và nút bấm để bạn chọn giọng ngay lập tức!\n"
        "3. <b>Đổi giọng nhanh:</b> Gõ <code>/voice tên_giọng</code> (ví dụ: <code>/voice hung_thinh</code>).\n\n"
        "<i>Lưu ý: Hệ thống xử lý tuần tự trên CPU để đảm bảo chất lượng giọng đọc ổn định nhất.</i>"
    )
    await update.message.reply_html(msg)


async def voices_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Lệnh /voices:
    Gửi danh sách giọng đọc kèm file âm thanh mẫu của từng giọng,
    kèm nút bấm 'Chọn giọng này' để người dùng đổi ngay chỉ với 1 click!
    """
    user_id = update.effective_user.id
    current_voice = user_voices.get(user_id, DEFAULT_VOICE)

    await update.message.reply_html(
        "🎙 <b>DANH SÁCH & MẪU GIỌNG ĐỌC KOKORO VIỆT NAM</b>\n\n"
        "<i>Đang gửi các mẫu giọng để bạn nghe thử... "
        "Bấm vào nút <b>'Chọn giọng này'</b> dưới mỗi mẫu để áp dụng!</i>"
    )

    for code, info in VOICES.items():
        label = info.get("label", code)
        is_cur = " ⭐ (Đang chọn)" if code == current_voice else ""
        sample_file = SAMPLE_DIR / f"{code}.wav"

        # Nút bấm inline để người dùng chọn giọng trực tiếp
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(f"✅ Chọn giọng {label}", callback_data=f"set_voice:{code}")]
        ])

        caption = f"🎙 <b>{label}</b> (<code>{code}</code>){is_cur}"

        if sample_file.exists():
            with open(sample_file, "rb") as f:
                await update.message.reply_voice(
                    voice=f,
                    caption=caption,
                    parse_mode="HTML",
                    reply_markup=keyboard,
                )
        else:
            await update.message.reply_html(
                f"• {caption}\n(Dùng lệnh: <code>/voice {code}</code>)",
                reply_markup=keyboard,
            )

        # Nghỉ ngắn giữa các tin nhắn để tuân thủ rate-limit của Telegram
        await asyncio.sleep(0.3)


async def voice_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý sự kiện khi người dùng bấm nút 'Chọn giọng này' dưới mẫu giọng."""
    query = update.callback_query
    await query.answer()

    data = query.data or ""
    if data.startswith("set_voice:"):
        requested = data.split(":", 1)[1]
        if requested in VOICES:
            user_voices[query.from_user.id] = requested
            label = VOICES[requested].get("label", requested)
            await query.message.reply_html(
                f"🎉 <b>Đã chọn thành công!</b>\n"
                f"Từ bây giờ bot sẽ đọc bằng giọng: <b>{label}</b> (<code>{requested}</code>).\n"
                f"Hãy gửi câu bạn muốn đọc thử ngay nhé!"
            )
        else:
            await query.message.reply_text("❌ Mã giọng không hợp lệ.")


async def voice_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Lệnh /voice <voice_name> để chọn giọng qua text."""
    user_id = update.effective_user.id

    if not context.args:
        current_voice = user_voices.get(user_id, DEFAULT_VOICE)
        voice_label = VOICES.get(current_voice, {}).get("label", current_voice)
        await update.message.reply_html(
            f"Giọng hiện tại của bạn: <b>{voice_label}</b> (<code>{current_voice}</code>)\n\n"
            f"• Để nghe thử mẫu tất cả giọng: Gõ <code>/voices</code>\n"
            f"• Để đổi giọng nhanh: Gõ <code>/voice [mã_giọng]</code> (ví dụ: <code>/voice hung_thinh</code>)"
        )
        return

    requested = context.args[0].strip().lower()
    if requested not in VOICES:
        await update.message.reply_text(
            f"❌ Không tìm thấy giọng '{requested}'. Gõ /voices để nghe thử và xem danh sách."
        )
        return

    user_voices[user_id] = requested
    label = VOICES[requested].get("label", requested)
    await update.message.reply_html(
        f"✅ Đã đổi sang giọng: <b>{label}</b> (<code>{requested}</code>)"
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Xử lý tin nhắn văn bản và tạo giọng đọc."""
    user_id = update.effective_user.id
    text = (update.message.text or "").strip()

    if not text:
        await update.message.reply_text("Vui lòng gửi nội dung văn bản.")
        return

    if len(text) > MAX_TEXT_LENGTH:
        await update.message.reply_text(
            f"⚠️ Đoạn văn quá dài ({len(text)} ký tự). "
            f"Để đảm bảo CPU không quá tải, vui lòng gửi dưới {MAX_TEXT_LENGTH} ký tự mỗi lần."
        )
        return

    voice = user_voices.get(user_id, DEFAULT_VOICE)
    voice_label = VOICES.get(voice, {}).get("label", voice)

    status_msg = await update.message.reply_text(
        f"⏳ Đang xử lý giọng đọc [{voice_label}]... Xin vui lòng đợi một chút."
    )

    async def keep_action_alive():
        try:
            while True:
                await update.effective_chat.send_action(ChatAction.RECORD_VOICE)
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass

    action_task = asyncio.create_task(keep_action_alive())

    try:
        # Dùng lock để xử lý tuần tự từng request, tránh nghẽn CPU máy chủ
        async with tts_lock:
            loop = asyncio.get_running_loop()
            audio_bytes, phonemes, elapsed = await loop.run_in_executor(
                None, _synthesize_sync, text, voice
            )

        action_task.cancel()

        try:
            await status_msg.delete()
        except Exception:
            pass

        audio_stream = io.BytesIO(audio_bytes)
        audio_stream.name = f"kokoro_{voice}_{int(time.time())}.wav"

        caption = f"🎙 <b>Giọng:</b> {voice_label} | ⏱ <b>Thời gian tạo:</b> {elapsed:.1f}s"
        await update.message.reply_voice(
            voice=audio_stream,
            caption=caption,
            parse_mode="HTML",
        )
        logger.info(f"Đã tạo thành công cho user {user_id} ({len(text)} ký tự, {elapsed:.2f}s)")

    except Exception as e:
        action_task.cancel()
        logger.exception("Lỗi trong quá trình tạo giọng đọc")
        try:
            await status_msg.edit_text(f"❌ Đã xảy ra lỗi khi tạo âm thanh: {str(e)}")
        except Exception:
            await update.message.reply_text(f"❌ Đã xảy ra lỗi: {str(e)}")


def main():
    """Khởi động Telegram Bot."""
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN_HERE":
        print("=" * 60)
        print("LỖI: Chưa có TELEGRAM_BOT_TOKEN!")
        print("Cách 1: Mở file bot.py và thay 'YOUR_TELEGRAM_BOT_TOKEN_HERE' bằng token của bạn.")
        print("Cách 2: Chạy lệnh trong terminal:")
        print("        export TELEGRAM_BOT_TOKEN=\"token_cua_ban\"")
        print("        ./run_bot.sh")
        print("=" * 60)
        sys.exit(1)

    print("=" * 60)
    print("Khởi động Kokoro Vietnamese Telegram Bot...")
    print(f"• Thiết bị xử lý: {DEVICE}")
    print(f"• Giọng mặc định: {DEFAULT_VOICE}")
    print("• Đang tải trước mô hình mặc định...")
    get_tts_engine(DEFAULT_VOICE)
    print("• Nạp mô hình hoàn tất! Đang kết nối tới Telegram...")
    print("=" * 60)

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Đăng ký các command
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("voices", voices_command))
    application.add_handler(CommandHandler("voice", voice_command))

    # Đăng ký callback query handler cho các nút bấm chọn giọng
    application.add_handler(CallbackQueryHandler(voice_callback_handler))

    # Đăng ký handler xử lý text
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text)
    )

    print("Bot đã sẵn sàng và đang lắng nghe tin nhắn... (Nhấn Ctrl+C để dừng)")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

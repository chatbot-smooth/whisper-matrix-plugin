from __future__ import annotations

import asyncio

from mautrix.client import Client
from mautrix.types import (
    EventType,
    MessageEvent,
    MessageType,
    TextMessageEventContent,
    EventID,
    RoomID,
)
import torch
import whisper
import numpy as np
import asyncio
import ffmpeg


class TranscriptorBot:

    user_id = "@transcriptorbot:example.com"
    base_url = "https://matrix.server.com"
    token = "syt_xyz"

    def __init__(self):
        # Create the client to access Matrix.
        self.client = Client(
            mxid=TranscriptorBot.user_id,
            base_url=TranscriptorBot.base_url,
            token=TranscriptorBot.token,
        )
        self.client.ignore_initial_sync = True
        self.client.ignore_first_sync = True

        self.client.add_event_handler(EventType.ROOM_MESSAGE, self.handle_message)

    async def handle_message(self, event: MessageEvent) -> None:
        if event.content.msgtype != MessageType.AUDIO:
            return

        audio_bytes = await self.client.download_media(url=event.content.url)

        print(f"We are transcribing the audio sent by {event.sender}...")

        asyncio.create_task(
            self.transcribe(
                audio_bytes=audio_bytes, room_id=event.room_id, event_id=event.event_id
            )
        )
        print(f"The audio sent by {event.sender} has been transcribed.")

    async def start(self):
        print("Starting Transcriptor")
        whoami = await self.client.whoami()
        print(f"\tConnected, I'm {whoami.user_id} using {whoami.device_id}")
        await self.client.start(None)

    async def transcribe(self, audio_bytes: bytes, room_id: RoomID, event_id: EventID):
        audio = self.load_audio(audio_bytes)

        audio = torch.from_numpy(audio)

        model = whisper.load_model("base")

        result = model.transcribe(audio, fp16=False)

        caption_content = TextMessageEventContent(
            msgtype=MessageType.TEXT,
            body=result.get("text", "Sorry!!").strip(),
        )

        caption_content.set_reply(event_id)

        await self.client.send_message(room_id=room_id, content=caption_content)

    def load_audio(self, file: str | bytes, sr: int = 16000):
        """
        Open an audio file and read as mono waveform, resampling as necessary

        Parameters
        ----------
        file: (str, bytes)
            The audio file to open or bytes of audio file

        sr: int
            The sample rate to resample the audio if necessary

        Returns
        -------
        A NumPy array containing the audio waveform, in float32 dtype.
        """

        if isinstance(file, bytes):
            inp = file
            file = "pipe:"
        else:
            inp = None

        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input(file, threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=inp)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


async def main():
    bot = TranscriptorBot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())

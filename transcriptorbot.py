from __future__ import annotations

import asyncio
import os

import ffmpeg
import numpy as np
import torch
from mautrix.client import Client
from mautrix.types import (EventType, Membership, MessageEvent, MessageType,
                           StrippedStateEvent, TextMessageEventContent, UserID)
from yarl import URL

import whisper

try:
    import dotenv
except ImportError:
    print("You must install the dotenv library")
    print("pip install python-dotenv")
    exit()


from dotenv import load_dotenv

load_dotenv()


class TranscriptorBot:

    user_id: UserID = os.getenv("MATRIX_USER_ID")
    base_url: URL = URL(os.getenv("MATRIX_BASE_URL"))
    token: str = os.getenv("MATRIX_TOKEN")

    def __init__(self):
        # Create the client to access Matrix.
        self.client = Client(
            mxid=self.user_id,
            base_url=self.base_url,
            token=self.token,
        )
        self.client.ignore_initial_sync = True
        self.client.ignore_first_sync = True

        self.model = whisper.load_model("base")

        # Register two handlers, one for room memberships (invites) and another for room messages.

        self.client.add_event_handler(EventType.ROOM_MEMBER, self.handle_invite)
        self.client.add_event_handler(EventType.ROOM_MESSAGE, self.handle_message)

    async def handle_message(self, event: MessageEvent) -> None:
        if event.content.msgtype != MessageType.AUDIO:
            return

        audio_bytes = await self.client.download_media(url=event.content.url)

        asyncio.create_task(self.transcribe(audio_bytes=audio_bytes, event=event))

    async def handle_invite(self, event: StrippedStateEvent) -> None:
        # Ignore the message if it's not an invitation for us.
        if (
            event.state_key == self.user_id
            and event.content.membership == Membership.INVITE
        ):
            # If it is, join the room.
            await self.client.join_room(event.room_id)

    async def start(self):
        print("Starting TranscriptorBot")
        whoami = await self.client.whoami()
        print(f"\tConnected, I'm {whoami.user_id} using {whoami.device_id}")
        await self.client.start(None)

    async def transcribe(self, audio_bytes: bytes, event: MessageEvent):

        print(f"We are transcribing the audio sent by {event.sender}...")

        audio = self.load_audio(audio_bytes)

        audio = torch.from_numpy(audio)

        result = self.model.transcribe(audio, fp16=False)

        caption_content = TextMessageEventContent(
            msgtype=MessageType.TEXT,
            body=result.get("text", "Sorry, I was unable to transcribe a message.").strip(),
        )

        caption_content.set_reply(event.event_id)

        await self.client.send_message(room_id=event.room_id, content=caption_content)

        print(f"The audio sent by {event.sender} has been transcribed.")

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

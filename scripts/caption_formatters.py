"""Caption formatters for ai-toolkit output mode."""

from abc import ABC, abstractmethod


class BaseCaptionFormatter(ABC):
    @abstractmethod
    def format(self, payload: dict) -> str:
        """Return a caption string for the given Qdrant payload dict."""
        ...


class NarrativeCaptionFormatter(BaseCaptionFormatter):
    """Merges scene, temporal events, and dialogue into a narrative caption."""

    def format(self, payload: dict) -> str:
        scene = payload.get("scene") or ""
        events = payload.get("events") or []
        transcription = payload.get("transcription") or []
        caption = payload.get("caption") or ""

        if not events:
            return caption

        sentences = self._build_narrative(events, transcription)
        if not sentences:
            return caption

        normalised = [s[0].lower() + s[1:] if s else s for s in sentences]
        narrative = ", ".join(normalised) + "."

        if scene:
            return f"{scene}. {narrative}"
        return narrative

    def _build_narrative(
        self,
        events: list[dict],
        transcription: list[dict],
    ) -> list[str]:
        used: set[int] = set()
        sentences: list[str] = []

        for event in events:
            e_start = float(event.get("start", 0))
            e_end = float(event.get("end", 0))
            description = (event.get("description") or "").strip()

            overlapping = [
                (i, t)
                for i, t in enumerate(transcription)
                if i not in used
                and float(t.get("start", 0)) < e_end
                and float(t.get("end", 0)) > e_start
            ]
            if overlapping:
                used.update(i for i, _ in overlapping)
                dialogue = " ".join(
                    (t.get("text") or "").strip() for _, t in overlapping
                ).strip()
                if dialogue and description:
                    sentences.append(f'{description} "{dialogue}"')
                elif dialogue:
                    sentences.append(f'"{dialogue}"')
                elif description:
                    sentences.append(description)
            elif description:
                sentences.append(description)

        return sentences


FORMATTERS: dict[str, type[BaseCaptionFormatter]] = {
    "narrative": NarrativeCaptionFormatter,
}

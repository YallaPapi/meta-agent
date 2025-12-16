"""Photo Manager - Handles mood-based photo selection.

Organizes photos in buckets by mood:
- happy/ - smiling, laughing
- sad/ - pouty, disappointed
- flirty/ - winking, cute poses
- casual/ - everyday moments
- morning/ - coffee, cozy
- night/ - going out, relaxed
- excited/ - enthusiastic
- selfie/ - generic
"""

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)

# Valid mood categories
MOODS = ["happy", "sad", "flirty", "casual", "morning", "night", "excited", "selfie"]

# Photo send probability (30% as per PRD)
PHOTO_SEND_PROBABILITY = 0.30


class PhotoManager:
    """Manages photo buckets and selection logic."""

    def __init__(self, base_path: str, persona_name: str = "mia"):
        """Initialize photo manager.

        Args:
            base_path: Base path to photos directory
            persona_name: Name of persona (subdirectory)
        """
        self.base_path = Path(base_path) / persona_name
        self._validate_buckets()

    def _validate_buckets(self):
        """Validate that photo bucket directories exist."""
        if not self.base_path.exists():
            logger.warning(f"Photo directory does not exist: {self.base_path}")
            logger.info("Creating directory structure...")
            self._create_bucket_structure()

    def _create_bucket_structure(self):
        """Create the mood bucket directories."""
        for mood in MOODS:
            mood_dir = self.base_path / mood
            mood_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created: {mood_dir}")

    def get_photos_for_mood(self, mood: str) -> List[Path]:
        """Get all photos for a given mood.

        Args:
            mood: Mood category

        Returns:
            List of photo paths
        """
        if mood not in MOODS:
            logger.warning(f"Unknown mood: {mood}, defaulting to 'selfie'")
            mood = "selfie"

        mood_dir = self.base_path / mood
        if not mood_dir.exists():
            return []

        # Get all image files
        extensions = {".jpg", ".jpeg", ".png", ".webp"}
        photos = [
            p for p in mood_dir.iterdir()
            if p.suffix.lower() in extensions
        ]
        return photos

    def select_photo(
        self,
        mood: str,
        time_override: bool = True,
        force_send: bool = False,
    ) -> Optional[str]:
        """Select a photo based on mood and context.

        Args:
            mood: Requested mood category
            time_override: Whether to apply time-of-day override
            force_send: If True, always return a photo (ignore probability)

        Returns:
            Path to selected photo, or None if not sending
        """
        # Check probability (unless forced)
        if not force_send and random.random() > PHOTO_SEND_PROBABILITY:
            logger.debug("Skipping photo (probability check)")
            return None

        # Apply time-of-day override
        if time_override:
            mood = self._apply_time_override(mood)

        # Get photos for mood
        photos = self.get_photos_for_mood(mood)

        if not photos:
            # Fallback to selfie
            logger.warning(f"No photos for mood '{mood}', trying 'selfie'")
            photos = self.get_photos_for_mood("selfie")

        if not photos:
            logger.error("No photos available in any bucket!")
            return None

        # Random selection
        selected = random.choice(photos)
        logger.info(f"Selected photo: {selected.name} (mood: {mood})")
        return str(selected)

    def _apply_time_override(self, mood: str) -> str:
        """Override mood based on time of day.

        - Before 10am: Use 'morning' for casual/selfie
        - After 10pm: Use 'night' for casual/selfie

        Args:
            mood: Original mood

        Returns:
            Possibly modified mood
        """
        hour = datetime.now().hour

        # Only override generic moods
        if mood not in ["casual", "selfie"]:
            return mood

        if hour < 10:
            logger.debug(f"Time override: {mood} -> morning (hour={hour})")
            return "morning"
        elif hour >= 22:
            logger.debug(f"Time override: {mood} -> night (hour={hour})")
            return "night"

        return mood

    def get_stats(self) -> dict:
        """Get statistics about photo buckets.

        Returns:
            Dict with count per mood
        """
        stats = {}
        for mood in MOODS:
            photos = self.get_photos_for_mood(mood)
            stats[mood] = len(photos)
        return stats


if __name__ == "__main__":
    # Test
    logging.basicConfig(level=logging.DEBUG)
    pm = PhotoManager("./photos", "mia")
    print("Stats:", pm.get_stats())

    # Test selection
    for mood in ["happy", "sad", "casual"]:
        photo = pm.select_photo(mood, force_send=True)
        print(f"{mood}: {photo}")

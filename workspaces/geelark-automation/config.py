"""Configuration for DM Bot.

Handles:
- Persona definitions
- Photo bucket paths
- LLM API keys
- Timing settings
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class PersonaConfig:
    """Persona character definition."""
    name: str = "Mia"
    age: int = 22
    occupation: str = "psychology student"
    university: str = "UCLA"
    personality: str = "friendly, curious, casual"
    texting_style: str = "short messages, occasional emojis, asks questions"
    goal: str = "learn their city naturally, then share a link"

    def to_system_prompt(self) -> str:
        """Generate system prompt from persona config."""
        return f"""You are {self.name}, a {self.age}-year-old {self.occupation} at {self.university}.

Personality: {self.personality}
Texting style: {self.texting_style}

Your goal: {self.goal}

IMPORTANT: Always respond with valid JSON in this exact format:
{{
  "text": "your message here",
  "send_photo": true or false,
  "photo_mood": "happy|sad|flirty|casual|morning|night|excited|selfie",
  "goal_status": "chatting|asking_location|got_location|sending_link|sent_link|rejected"
}}

Photo guidelines:
- Send photos ~30% of the time to seem natural
- "sad" when they decline something or seem distant
- "excited" when they show interest
- "flirty" when conversation gets playful
- "morning"/"night" based on time references in chat
- "casual" for normal conversation

Goal progression:
- Start with "chatting" - build rapport
- Move to "asking_location" - naturally ask where they're from
- "got_location" - once they mention their city
- "sending_link" - transition to sharing your link
- "sent_link" - after you've shared it
- "rejected" - if they decline, back off gracefully

Keep responses SHORT (1-2 sentences). Sound like you're texting a friend.
Never sound salesy or robotic. Be genuinely curious about them."""


@dataclass
class DMBotConfig:
    """Main configuration for DM bot."""
    # Persona
    persona: PersonaConfig = field(default_factory=PersonaConfig)
    persona_name: str = "mia"  # For photo bucket

    # Paths
    photo_bucket_path: str = "./photos"
    state_dir: str = "./state"

    # External website
    external_url: str = "https://example.com"

    # LLM settings
    llm_provider: str = "ollama"  # "ollama", "anthropic", or "openai"
    llm_api_key: str = ""
    llm_model: str = "qwen2.5:7b"  # Default to qwen2.5 for Ollama
    ollama_host: str = "http://localhost:11434"

    # Timing
    min_response_delay: float = 2.0  # seconds
    max_response_delay: float = 8.0
    poll_interval: float = 30.0  # seconds between inbox checks
    photo_probability: float = 0.30

    # Geelark / Device
    device_id: Optional[str] = None
    appium_host: str = "127.0.0.1"
    appium_port: int = 4723


def load_config(config_path: str = "config.yaml") -> DMBotConfig:
    """Load configuration from YAML file and environment.

    Args:
        config_path: Path to YAML config file

    Returns:
        DMBotConfig instance
    """
    config = DMBotConfig()

    # Load from YAML if exists
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)

        if data:
            # Persona
            if "persona" in data:
                config.persona = PersonaConfig(**data["persona"])
                config.persona_name = data["persona"].get("name", "mia").lower()

            # Paths
            config.photo_bucket_path = data.get("photo_bucket_path", config.photo_bucket_path)
            config.state_dir = data.get("state_dir", config.state_dir)
            config.external_url = data.get("external_url", config.external_url)

            # LLM
            config.llm_provider = data.get("llm_provider", config.llm_provider)
            config.llm_model = data.get("llm_model", config.llm_model)
            config.ollama_host = data.get("ollama_host", config.ollama_host)

            # Timing
            config.poll_interval = data.get("poll_interval", config.poll_interval)
            config.photo_probability = data.get("photo_probability", config.photo_probability)

            # Device
            config.device_id = data.get("device_id", config.device_id)

    # Override with environment variables
    config.llm_api_key = os.environ.get("ANTHROPIC_API_KEY", "") or \
                         os.environ.get("OPENAI_API_KEY", "")

    return config


def save_default_config(config_path: str = "config.yaml"):
    """Save default configuration as YAML template."""
    default = {
        "persona": {
            "name": "Mia",
            "age": 22,
            "occupation": "psychology student",
            "university": "UCLA",
            "personality": "friendly, curious, casual",
            "texting_style": "short messages, occasional emojis, asks questions",
            "goal": "learn their city naturally, then share a link",
        },
        "photo_bucket_path": "./photos",
        "state_dir": "./state",
        "external_url": "https://example.com/your-link-here",
        "llm_provider": "ollama",  # "ollama", "anthropic", or "openai"
        "llm_model": "qwen2.5:7b",  # For Ollama; use "claude-3-5-sonnet-20241022" for Anthropic
        "ollama_host": "http://localhost:11434",
        "poll_interval": 30,
        "photo_probability": 0.30,
        "device_id": None,
    }

    with open(config_path, "w") as f:
        yaml.dump(default, f, default_flow_style=False, sort_keys=False)

    print(f"Saved default config to {config_path}")


if __name__ == "__main__":
    # Generate default config
    save_default_config()

    # Test loading
    config = load_config()
    print(f"Persona: {config.persona.name}, {config.persona.age}")
    print(f"LLM: {config.llm_provider} / {config.llm_model}")
    print("\nSystem prompt preview:")
    print(config.persona.to_system_prompt()[:500] + "...")

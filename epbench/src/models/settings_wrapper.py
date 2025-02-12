from pydantic_settings import BaseSettings

class SettingsWrapper(BaseSettings):
    # parameters from the .env variables with missing default values
    PROXY: dict = {}
    OPENAI_API_KEY : str = ''
    ANTHROPIC_API_KEY: str = ''
    REPLICATE_API_TOKEN: str = ''
    OPENROUTER_API_KEY: str = ''
    GOOGLE_API_KEY: str = ''
    DEEPSEEK_API_KEY: str = ''

    class Config:
        env_file = '.env' # default location, can be overridden
        env_file_encoding = "utf-8"

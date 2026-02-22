from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    postgres_url: str
    openrouter_api_key: str
    llm_model: str = "mistralai/mistral-7b-instruct"
    embedding_backend: str = "openai"

    class Config:
        env_file = ".env"

settings = Settings()

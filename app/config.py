from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    AWS_AK: str
    AWS_SAK: str
    openai_api_key: str

    class Config:
        env_file = '.env'

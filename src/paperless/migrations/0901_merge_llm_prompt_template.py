from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("paperless", "0013_applicationconfiguration_llm_request_timeout"),
        ("paperless", "0900_applicationconfiguration_llm_prompt_template"),
    ]

    operations = []

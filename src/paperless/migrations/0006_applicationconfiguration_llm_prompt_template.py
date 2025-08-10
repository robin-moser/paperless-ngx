# Generated manually

from django.db import migrations, models
import django.utils.translation


class Migration(migrations.Migration):

    dependencies = [
        ('paperless', '0005_applicationconfiguration_ai_enabled_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='applicationconfiguration',
            name='llm_prompt_template',
            field=models.TextField(
                blank=True,
                help_text=django.utils.translation.gettext_lazy(
                    'Custom prompt template for AI classification. Available placeholders: '
                    '{filename}, {content}, {available_tags}, {available_correspondents}'
                ),
                null=True,
                verbose_name=django.utils.translation.gettext_lazy('Custom AI prompt template'),
            ),
        ),
    ]
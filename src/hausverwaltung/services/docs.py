from __future__ import annotations
import os
from jinja2 import Template

# Optional LLM-„Veredelung“
def refine_with_llm(text: str, system_prompt: str = "Formuliere höflich, klar und präzise auf Deutsch.") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return text
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":text},
            ],
        )
        return resp.choices[0].message.content or text
    except Exception:
        return text

def render_letter(template_str: str, **kwargs) -> str:
    tpl = Template(template_str)
    return tpl.render(**kwargs)

TEMPLATE_REMINDER = """\
Betreff: Zahlungserinnerung – Wohnung {{ unit_no }}, {{ street }}, {{ zip }} {{ city }}

Sehr geehrte/r {{ tenant_name }},

für den Mietzeitraum {{ period }} ist bisher kein Zahlungseingang verzeichnet.
Offener Betrag: {{ amount }} €.

Bitte überweisen Sie den Betrag bis spätestens {{ due_date }} auf das bekannte Konto.

Mit freundlichen Grüßen
{{ owner_name }}
"""
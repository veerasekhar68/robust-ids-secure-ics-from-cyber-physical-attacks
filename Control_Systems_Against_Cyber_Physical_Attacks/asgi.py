"""
ASGI config for Control_Systems_Against_Cyber_Physical_Attacks project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
"""

import os

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Control_Systems_Against_Cyber_Physical_Attacks.settings')

application = get_asgi_application()

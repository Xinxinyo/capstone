from django.contrib import admin

# Register your models here.

from .models import User
from .models import Item
from .models import Hub17
from .models import Hub17_Result

admin.site.register(User)
admin.site.register(Item)
admin.site.register(Hub17)
admin.site.register(Hub17_Result)
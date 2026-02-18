import os
from django.conf import settings
from django.shortcuts import render
import pandas as pd
from users.forms import UserRegistrationForm


def index(request):
    return render(request, 'index.html', {})

def AdminLogin(request):
    return render(request, 'AdminLogin.html', {})

def UserLogin(request):
    print("Rendering UserLogin page...")  # Check if view is called
    return render(request, 'UserLogin.html')


def UserRegister(request):
    form = UserRegistrationForm()
    return render(request, 'UserRegistration.html', {'form': form})

from datetime import datetime
import socket
from django.conf import settings
from django.core.paginator import Paginator, EmptyPage, PageNotAnInteger
from users.forms import  UserRegistrationForm
from django.shortcuts import render, redirect
from django.contrib import messages
import pandas as pd
import os
from .models import UserRegistrationModel
from django.contrib import messages
from .models import UserRegistrationModel


def base(request):
    return render(request,'base.html')

# Add these views to your views.py file

def user_profile(request):
    """View to display user profile"""
    if 'id' not in request.session:
        messages.error(request, 'Please login to view your profile.')
        return redirect('UserLoginCheck')
    
    try:
        user = UserRegistrationModel.objects.get(id=request.session['id'])
        context = {
            'user': user
        }
        return render(request, 'user_profile.html', context)
    except UserRegistrationModel.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('UserLoginCheck')


def update_profile(request):
    """View to update user profile"""
    if 'id' not in request.session:
        messages.error(request, 'Please login to update your profile.')
        return redirect('UserLoginCheck')
    
    try:
        user = UserRegistrationModel.objects.get(id=request.session['id'])
        
        if request.method == 'POST':
            # Get form data
            name = request.POST.get('name')
            email = request.POST.get('email')
            mobile = request.POST.get('mobile')
            locality = request.POST.get('locality')
            password = request.POST.get('password')
            
            # Validate email uniqueness (excluding current user)
            if UserRegistrationModel.objects.filter(email=email).exclude(id=user.id).exists():
                messages.error(request, 'Email already exists.')
                return render(request, 'update_profile.html', {'user': user})
            
            # Validate mobile uniqueness (excluding current user)
            if UserRegistrationModel.objects.filter(mobile=mobile).exclude(id=user.id).exists():
                messages.error(request, 'Mobile number already exists.')
                return render(request, 'update_profile.html', {'user': user})
            
            # Update user data
            user.name = name
            user.email = email
            user.mobile = mobile
            user.locality = locality
            
            # Only update password if provided
            if password:
                user.password = password
            
            user.save()
            
            # Update session name if changed
            request.session['loggeduser'] = name
            
            messages.success(request, 'Profile updated successfully!')
            return redirect('user_profile')
        
        context = {
            'user': user
        }
        return render(request, 'update_profile.html', context)
        
    except UserRegistrationModel.DoesNotExist:
        messages.error(request, 'User not found.')
        return redirect('UserLoginCheck')



def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'You have been successfully registered') 
            return render(request,'UserRegistration.html')
        else:
            messages.error(request, 'Email or Mobile Already Exists')
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistration.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('password')  # Corrected key to match the form input name
        try:
            user = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            if user.status == "activated":
                request.session['id'] = user.id
                request.session['loggeduser'] = user.name
                return redirect('UserHome')  # Redirect to User Home after successful login
            else:
                messages.error(request, 'Your account is not activated.')
        except UserRegistrationModel.DoesNotExist:
            messages.error(request, 'Invalid Login ID or Password')
    return render(request, 'UserLogin.html')  # Ensure this is the correct template name


def UserHome(request):
    return render(request, 'UserHome.html')



import time
import psutil
from django.shortcuts import render
from django.http import HttpResponse
from django.template.loader import get_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from xhtml2pdf import pisa

# ✅ CSV file path
CSV_PATH = settings.MEDIA_ROOT / "Dataset.csv"

# ✅ Load dataset
def load_dataset():
    df = pd.read_csv(CSV_PATH)
    df.dropna(inplace=True)

    label_col = 'IT_B_Label'
    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' not found in dataset.")

    print("\n[INFO] Label distribution:")
    print(df[label_col].value_counts())

    y = df[label_col]
    X = df.drop(columns=['IT_B_Label', 'IT_M_Label', 'NST_B_Label', 'NST_M_Label'], errors='ignore')

    # Only numeric features
    X = X.select_dtypes(include=['number'])

    return X, y

# ✅ Intrusion Detection Logic (Hybrid: LightGBM + Isolation Forest)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def run_hybrid_ids(X, y):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    # SMOTE
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    # Start CPU/time
    start_time = time.time()
    start_cpu = psutil.cpu_percent(interval=None)

    # LightGBM
    clf = LGBMClassifier(
        n_estimators=50,
        class_weight='balanced',
        max_depth=6,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred_lgbm = clf.predict(X_test)

    # Isolation Forest
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_train)
    y_pred_iso = iso.predict(X_test)
    y_pred_iso = [0 if v == -1 else 1 for v in y_pred_iso]

    # Hybrid decision (AND rule)
    y_pred_combined = [
        1 if (l == 1 and i == 1) else 0
        for l, i in zip(y_pred_lgbm, y_pred_iso)
    ]

    # End CPU/time
    end_time = time.time()
    end_cpu = psutil.cpu_percent(interval=None)

    # Metrics
    acc = accuracy_score(y_test, y_pred_combined)
    prec = precision_score(y_test, y_pred_combined, zero_division=0)
    rec = recall_score(y_test, y_pred_combined, zero_division=0)
    f1 = f1_score(y_test, y_pred_combined, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_combined)

    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    detection_rate = tp / (tp + fn) if (tp + fn) else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0

    print("\n[INFO] Confusion Matrix:\n", cm)
    print("\n[INFO] Classification Report:\n",
          classification_report(y_test, y_pred_combined))

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "confusion_matrix": cm.tolist(),

        "detection_rate": round(detection_rate * 100, 2),
        "false_positive_rate": round(false_positive_rate * 100, 2),
        "response_time": round(end_time - start_time, 2),
        "cpu_usage": round((start_cpu + end_cpu) / 2, 2),
    }
# ✅ Home Page
def home(request):
    return render(request, 'userhome.html')
def userhome(request):
    return render(request,'UserHome.html')

# ✅ Detection Result
def detect_intrusion(request):
    try:
        X, y = load_dataset()
        result = run_hybrid_ids(X, y)
        return render(request, 'result.html', {'result': result})
    except Exception as e:
        return HttpResponse(f"<h3>Error: {str(e)}</h3>")

# ✅ PDF Report Download
def download_pdf(request):
    try:
        # ML pipeline
        X, y = load_dataset()
        result = run_hybrid_ids(X, y)

        # Expected structure from run_hybrid_ids
        # result = {
        #   'accuracy': 0.85,
        #   'precision': 0.72,
        #   'recall': 0.79,
        #   'f1_score': 0.75,
        #   'confusion_matrix': [[7712, 1100], [742, 2854]],
        #   'report': report_data
        # }

        # System info
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        system_ip = socket.gethostbyname(socket.gethostname())
        user = request.user.username if request.user.is_authenticated else "Anonymous"
        session_id = request.session.session_key or "N/A"

        template = get_template("pdf_template.html")
        html = template.render({
            "accuracy": result["accuracy"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1_score": result["f1_score"],
            "cm": result["confusion_matrix"],
            "report": result.get("report"),
            "timestamp": timestamp,
            "system_ip": system_ip,
            "user": user,
            "session_id": session_id,
        })

        response = HttpResponse(content_type="application/pdf")
        response["Content-Disposition"] = 'attachment; filename="ICS_IDS_Report.pdf"'

        pisa_status = pisa.CreatePDF(html, dest=response)

        if pisa_status.err:
            return HttpResponse("PDF Generation Error: Invalid HTML/CSS")

        return response

    except Exception as e:
        return HttpResponse(f"<h3>PDF Generation Error: {str(e)}</h3>")

def sender_details(request):
    """View to display sender-related network details with pagination"""
    
    # Path to your CSV file
    csv_path = os.path.join(settings.BASE_DIR, 'media', 'Dataset.csv')
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Select sender-related columns
        sender_columns = [
            'sAddress', 'sMACs', 'sIPs', 'sPackets', 'sBytesSum', 
            'sBytesMax', 'sBytesMin', 'sBytesAvg', 'sLoad', 
            'sPayloadSum', 'sPayloadMax', 'sPayloadMin', 'sPayloadAvg',
            'sInterPacketAvg', 'sttl', 'sAckRate', 'sUrgRate', 
            'sFinRate', 'sPshRate', 'sSynRate', 'sRstRate', 
            'sWinTCP', 'sFragmentRate', 'sAckDelayMax', 'sAckDelayMin', 
            'sAckDelayAvg', 'protocol', 'startDate', 'duration'
        ]
        
        # Filter only sender columns that exist in the dataframe
        available_sender_cols = [col for col in sender_columns if col in df.columns]
        sender_df = df[available_sender_cols]
        
        # Convert to list of dictionaries for template
        sender_data = sender_df.to_dict('records')
        
        # Get unique sender addresses for filtering
        unique_senders = df['sAddress'].unique().tolist() if 'sAddress' in df.columns else []
        
        # Calculate some statistics
        stats = {
            'total_records': len(sender_df),
            'unique_senders': len(unique_senders),
            'avg_packets': round(df['sPackets'].mean(), 2) if 'sPackets' in df.columns else 0,
            'avg_bytes': round(df['sBytesSum'].mean(), 2) if 'sBytesSum' in df.columns else 0,
        }
        
        # Pagination
        page = request.GET.get('page', 1)
        items_per_page = request.GET.get('items_per_page', 25)  # Default 25 items per page
        
        try:
            items_per_page = int(items_per_page)
            # Limit items per page to reasonable values
            if items_per_page < 10:
                items_per_page = 10
            elif items_per_page > 100:
                items_per_page = 100
        except (ValueError, TypeError):
            items_per_page = 25
        
        paginator = Paginator(sender_data, items_per_page)
        
        try:
            sender_page = paginator.page(page)
        except PageNotAnInteger:
            sender_page = paginator.page(1)
        except EmptyPage:
            sender_page = paginator.page(paginator.num_pages)
        
        context = {
            'sender_data': sender_page,
            'unique_senders': unique_senders,
            'stats': stats,
            'items_per_page': items_per_page,
            'total_pages': paginator.num_pages,
            'current_page': sender_page.number,
        }
        
        return render(request, 'sender_details.html', context)
        
    except FileNotFoundError:
        context = {
            'error': 'CSV file not found. Please upload the network data file.',
            'sender_data': [],
            'stats': {}
        }
        return render(request, 'sender_details.html', context)
    except Exception as e:
        context = {
            'error': f'Error loading data: {str(e)}',
            'sender_data': [],
            'stats': {}
        }
        return render(request, 'sender_details.html', context)


def receiver_details(request):
    """View to display receiver-related network details with pagination"""
    
    # Path to your CSV file
    csv_path = os.path.join(settings.BASE_DIR, 'media', 'Dataset.csv')
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Select receiver-related columns
        receiver_columns = [
            'rAddress', 'rMACs', 'rIPs', 'rPackets', 'rBytesSum', 
            'rBytesMax', 'rBytesMin', 'rBytesAvg', 'rLoad', 
            'rPayloadSum', 'rPayloadMax', 'rPayloadMin', 'rPayloadAvg',
            'rInterPacketAvg', 'rttl', 'rAckRate', 'rUrgRate', 
            'rFinRate', 'rPshRate', 'rSynRate', 'rRstRate', 
            'rWinTCP', 'rFragmentRate', 'rAckDelayMax', 'rAckDelayMin', 
            'rAckDelayAvg', 'protocol', 'endDate', 'duration'
        ]
        
        # Filter only receiver columns that exist in the dataframe
        available_receiver_cols = [col for col in receiver_columns if col in df.columns]
        receiver_df = df[available_receiver_cols]
        
        # Convert to list of dictionaries for template
        receiver_data = receiver_df.to_dict('records')
        
        # Get unique receiver addresses for filtering
        unique_receivers = df['rAddress'].unique().tolist() if 'rAddress' in df.columns else []
        
        # Calculate some statistics
        stats = {
            'total_records': len(receiver_df),
            'unique_receivers': len(unique_receivers),
            'avg_packets': round(df['rPackets'].mean(), 2) if 'rPackets' in df.columns else 0,
            'avg_bytes': round(df['rBytesSum'].mean(), 2) if 'rBytesSum' in df.columns else 0,
        }
        
        # Pagination
        page = request.GET.get('page', 1)
        items_per_page = request.GET.get('items_per_page', 25)  # Default 25 items per page
        
        try:
            items_per_page = int(items_per_page)
            # Limit items per page to reasonable values
            if items_per_page < 10:
                items_per_page = 10
            elif items_per_page > 100:
                items_per_page = 100
        except (ValueError, TypeError):
            items_per_page = 25
        
        paginator = Paginator(receiver_data, items_per_page)
        
        try:
            receiver_page = paginator.page(page)
        except PageNotAnInteger:
            receiver_page = paginator.page(1)
        except EmptyPage:
            receiver_page = paginator.page(paginator.num_pages)
        
        context = {
            'receiver_data': receiver_page,
            'unique_receivers': unique_receivers,
            'stats': stats,
            'items_per_page': items_per_page,
            'total_pages': paginator.num_pages,
            'current_page': receiver_page.number,
        }
        
        return render(request, 'receiver_details.html', context)
        
    except FileNotFoundError:
        context = {
            'error': 'CSV file not found. Please upload the network data file.',
            'receiver_data': [],
            'stats': {}
        }
        return render(request, 'receiver_details.html', context)
    except Exception as e:
        context = {
            'error': f'Error loading data: {str(e)}',
            'receiver_data': [],
            'stats': {}
        }
        return render(request, 'receiver_details.html', context)

def prediction_page(request):
    """
    Displays the prediction page with a button to run the IDS detection.
    """
    return render(request, 'prediction.html')


import pandas as pd

pd.set_option('display.max_columns', None)
df = pd.read_csv("ProyekHappines/Mental_Health_and_Social_Media_Balance_Dataset.csv")
df.info()
print(df['Social_Media_Platform'].unique())
print("Kita coba Gunakan Matrik Korelasi untuk antar kolom")
df= df.drop(columns="User_ID")


# Tampilkan kolom-kolom baru untuk verifikasii

print(df.info())
correlation_matrix =  df.corr(numeric_only=True)
print(correlation_matrix)

happines_mapping = {
    1:"Extremely Unhappy",
    2:"Very Unhappy",
    3:"Unhappy",
    4:"Slightly Unhappy",
    5:"Neutral",
    6:"Slightly Happy",
    7:"Happy",
    8:"Very Happy",
    9:"Extremely Happy",
    10:"Fully Satisfied"
}
df['Happiness_level_label'] = df['Happiness_Index(1-10)'].map(happines_mapping)
print("Hapiness scale sudah selesai")
df.info()
print(df.describe())
print(df.head(10))

#Visualisasi
import matplotlib.pyplot as plt
print(df['Happiness_level_label'].value_counts())
Fully_Satisfied = df[df['Happiness_level_label']=="Fully Satisfied"]
Very_Happy = df[df['Happiness_level_label']=="Very Happy"]
Extremely_Happy = df[df['Happiness_level_label']=="Extremely Happy"]
Happy = df[df['Happiness_level_label']=="Happy"]
Slightly_Happy = df[df['Happiness_level_label']=="Slightly Happy"]
Neutral = df[df['Happiness_level_label']=="Neutral"]
Slightly_Unhappy = df[df['Happiness_level_label']=="Slightly Unhappy"]

plt.figure(figsize=(6,5))
plt.scatter(Fully_Satisfied['Stress_Level(1-10)'], Fully_Satisfied['Daily_Screen_Time(hrs)'], alpha=0.3, color='blue', label= "Fully Satisfied")
plt.scatter(Very_Happy['Stress_Level(1-10)'], Very_Happy['Daily_Screen_Time(hrs)'], alpha=0.3, color='orange', label="Very Happy")
plt.scatter(Extremely_Happy['Stress_Level(1-10)'], Extremely_Happy['Daily_Screen_Time(hrs)'], alpha=0.3, color='red', label="Very Happy")
plt.scatter(Happy['Stress_Level(1-10)'], Happy['Daily_Screen_Time(hrs)'], alpha=0.3, color='yellow', label = "Happy")
plt.scatter(Slightly_Happy['Stress_Level(1-10)'], Slightly_Happy['Daily_Screen_Time(hrs)'], alpha=0.3, color='grey', label="Slightly Happy")
plt.scatter(Neutral['Stress_Level(1-10)'], Neutral['Daily_Screen_Time(hrs)'], alpha=0.3, color='green', label="Neutral")
plt.scatter(Slightly_Unhappy['Stress_Level(1-10)'], Slightly_Unhappy['Daily_Screen_Time(hrs)'], alpha=0.3, color='black', label="Slightly Unhappy")
plt.xlabel("Level Stress")
plt.ylabel("Jumlah Screen Time")
plt.title("Jumlah Screen Time Daily vs Level Stress")
plt.legend()
print("======== Setelah dianalisis awal, Screen time yang tinggi cenderung Tingkat stressnya tinggi ========")


# Membangun Model Machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

x = df[['Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 'Stress_Level(1-10)', 'Days_Without_Social_Media', 'Exercise_Frequency(week)', 'Social_Media_Platform']]
y = df['Happiness_level_label']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
) # random state gunanya mengacak agar ada urutan 

numeric_kolom = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 'Stress_Level(1-10)', 'Days_Without_Social_Media', 'Exercise_Frequency(week)']
categorical_kolom = ['Gender', 'Social_Media_Platform']

preprocessing = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_kolom),
        ("ohe", OneHotEncoder(), categorical_kolom)
    ]
)
model = Pipeline(
    steps=[
        ("preprocessing", preprocessing),
        ("model", RandomForestClassifier())
    ]
)

model.fit(x_train, y_train) #Model Belajar
y_pred = model.predict(x_test) #Soal Ulangan
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


data_baru = pd.DataFrame([[40, "Male", 3, 7.0, 6.0, 2.0, 5.0, "Facebook"]], 
                         columns=['Age', 'Gender', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 'Stress_Level(1-10)', 'Days_Without_Social_Media', 'Exercise_Frequency(week)', 'Social_Media_Platform'])

print(model.predict(data_baru)[0])
print(model.predict_proba(data_baru)[0])

prediksi = model.predict(data_baru)[0]
presentase = max(model.predict_proba(data_baru)[0])
print(f"Model memprediksi {prediksi} dengan tingkat keyakina {presentase*100:.2f}%")

# Mulai membangun website
import joblib
joblib.dump(model, "model_happines_indeks")
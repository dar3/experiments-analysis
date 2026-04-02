import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

fe_load = pd.read_csv('../data/stroke_risk_dataset_v2.csv')


print("Krotki przeglad danych z datasetu:")
print(f"Przeanalizowano dane dla: {fe_load.shape[0]} pacjentów")
print(f"Średni wiek w badanej grupie wynosi: {fe_load['age'].mean():.1f} lat (od {fe_load['age'].min()} do {fe_load['age'].max()} lat).")
print(f"Średnie szacowane ryzyko udaru wynosi: {fe_load['stroke_risk_percentage'].mean():.1f}%.")
print(f"Odsetek pacjentów sklasyfikowanych jako 'zagrożeni' (at_risk=1): {(fe_load['at_risk'].mean() * 100):.1f}%\n")
print("Elementy najbardziej zwiększające ryzyko udaru:")
# one hot encoding
c_d = fe_load.copy()
c_d['gender'] = c_d['gender'].map({'Male': 1, 'Female': 0})

correl = c_d.corr()['stroke_risk_percentage'].sort_values(ascending=False)
correl = correl.drop(['stroke_risk_percentage', 'at_risk'])

print("Z analizy korelacji statystycznej wynika, że najważniejszymi czynnikami współwystępującymi z wysokim ryzykiem udaru są:")
for feat, corr_value in correl.head(5).items():
    print(f" -> {feat} (siła korelacji: {corr_value:.2f})")
print("\n")

print("Analiza najczęstszych objawów w próbce:")
symptoms = ['chest_pain', 'high_blood_pressure', 'irregular_heartbeat',
            'shortness_of_breath', 'fatigue_weakness', 'dizziness',
            'swelling_edema', 'neck_jaw_pain', 'excessive_sweating',
            'persistent_cough', 'nausea_vomiting', 'chest_discomfort',
            'cold_hands_feet', 'snoring_sleep_apnea', 'anxiety_doom']

st_freq = fe_load[symptoms].mean() * 100
st_freq = st_freq.sort_values(ascending=False)
print("5 najczęściej zgłaszanych dolegliwości to:")
for smpt, freq in st_freq.head(5).items():
    print(f"\n{smpt}: występuje u {freq:.1f}% badanych.")
print("\n")
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 6))
sns.histplot(data=fe_load, x='age', hue='at_risk', multiple="stack", bins=30, palette=['#4daf4a', '#e41a1c'])
plt.title('Rozkład wieku vs status ryzyka udaru (0 Bezpieczny, 1  Zagrożony)', fontsize=12)
plt.xlabel('Wiek')
plt.ylabel('Liczba pacjentów')
plt.tight_layout()
plt.savefig('w1_wiek_ryzyko.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
sns.kdeplot(data=fe_load, x='stroke_risk_percentage', hue='gender', fill=True, palette='Set2')
plt.title('Gęstość rozkładu ryzyka udaru w podziale na płeć', fontsize=12)
plt.xlabel('Szacowane ryzyko udaru (%)')
plt.ylabel('Gęstość')
plt.tight_layout()
plt.savefig('w2_gestosc_ryzyko.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
top_10_symptoms = st_freq.head(10)
sns.barplot(x=top_10_symptoms.values, y=top_10_symptoms.index, palette='mako')
plt.title('10 najczęstszych objawów w badanej grupie', fontsize=12)
plt.xlabel('Częstość występowania (%)')
plt.ylabel('Nazwa objawu')
plt.tight_layout()
plt.savefig('w3_top10_objawow.png', dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
top_features = correl.head(6).index.tolist() + ['stroke_risk_percentage']
sns.heatmap(c_d[top_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Macierz korelacji kluczowych czynników', fontsize=12)
plt.tight_layout()
plt.savefig('w4_heatmapa.png', dpi=300)
plt.close()
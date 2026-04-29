import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


df = pd.read_csv('../data/stroke_risk_dataset_v2.csv')


test_var = df['stroke_risk_percentage']
hypertension_grp = df[df['high_blood_pressure'] == 1]['stroke_risk_percentage'].dropna()
healthy_grp = df[df['high_blood_pressure'] == 0]['stroke_risk_percentage'].dropna()

print("Sprawdzenie normalnosci rozkladu przez stosowaniem testu T-Studenta")

print("Test Shapiro-wilka")

stat_hypertension, p_hypertension = stats.shapiro(hypertension_grp)
stat_healthy, p_healthy = stats.shapiro(healthy_grp)

print(f"Grupa z nadciśnieniem (N={len(hypertension_grp)}): Statystyka W = {stat_hypertension:.4f}, p-value = {p_hypertension:.2e}")
print(f"Grupa bez nadciśnienia (N={len(healthy_grp)}): Statystyka W = {stat_healthy:.4f}, p-value = {p_healthy:.2e}\n")


if p_hypertension < 0.05 or p_healthy < 0.05:
    print("p-value < 0.05. Odrzucamy hipotezę o normalności rozkładu.")
else:
    print("p-value >= 0.05. Brak podstaw do odrzucenia hipotezy o normalności.")



fig, axes = plt.subplots(1, 2, figsize=(14, 6))

stats.probplot(hypertension_grp, dist="norm", plot=axes[0])
axes[0].set_title('Q-Q Plot: Pacjenci z nadciśnieniem', fontsize=12)
axes[0].set_ylabel('Wartości z próby')
axes[0].set_xlabel('Kwantyle teoretyczne rozkładu normalnego')

stats.probplot(healthy_grp, dist="norm", plot=axes[1])
axes[1].set_title('Q-Q Plot: Pacjenci bez nadciśnienia', fontsize=12)
axes[1].set_ylabel('Wartości z próby')
axes[1].set_xlabel('Kwantyle teoretyczne rozkładu normalnego')

plt.tight_layout()
plt.savefig('qq_plot_norm.png', dpi=300)


plt.show()


print("Estymacja punktowa")
srednia_punktowa = test_var.mean()
odchylenie_punktowe = test_var.std(ddof=1)
print(f"Estymator punktowy średniego ryzyka udaru w całej próbie: {srednia_punktowa:.2f}%")
print(f"Estymator punktowy odchylenia standardowego: {odchylenie_punktowe:.2f}%\n")



print("Estymacja przedziałowa")
conf_lvl = 0.95
n = len(test_var)
std_err = stats.sem(test_var)


conf_interval = stats.t.interval(
    confidence=conf_lvl,
    df=n-1,
    loc=srednia_punktowa,
    scale=std_err
)
print(f"95% Przedział Ufności dla średniego ryzyka udaru: ({conf_interval[0]:.2f}%, {conf_interval[1]:.2f}%)\n")



print("Sprawdzanie hipotezy statystycznej")
print("Badanie wpływu nadciśnienia na ryzyko udaru.")
print("H0 (Hipoteza zerowa): Średnie ryzyko udaru u osób z nadciśnieniem i bez nadciśnienia jest takie samo.")
print("H1 (Hipoteza alternatywna): Średnie ryzyko udaru w obu grupach jest różne.\n")

# test T-Welcha dla 2 niezaleznych grup
t_stat, p_value = stats.ttest_ind(hypertension_grp.dropna(), healthy_grp.dropna(), equal_var=False)

print(f"Średnie ryzyko - grupa z nadciśnieniem: {hypertension_grp.mean():.2f}%")
print(f"Średnie ryzyko - grupa bez nadciśnienia: {healthy_grp.mean():.2f}%")
print(f"Wartość statystyki testowej t: {t_stat:.4f}")
print(f"Wartość p-value: {p_value:.4g}")



alfa = 0.05
if p_value < alfa:
    print("\nOdrzucamy hipotezę zerową na korzyść hipotezy alternatywnej.")
    print("Różnica między grupami jest statystycznie istotna.")
else:
    print("\nBrak podstaw do odrzucenia hipotezy zerowej.")
    print("Różnica między grupami nie jest statystycznie istotna (może np. wynikać z przypadku).")
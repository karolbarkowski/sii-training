import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(42)

# ── Środowisko: siatka 8x8, start w lewym górnym rogu, cel w prawym dolnym ──
ROZMIAR = 8
START = (0, 0)
CEL = (7, 7)
PRZESZKODY = {(1, 3), (2, 3), (3, 3), (4, 3), (5, 5), (6, 5), (3, 6), (4, 6)}
RUCHY = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # góra, dół, lewo, prawo


def krok(stan, akcja):
    """Zwraca (nowy_stan, nagroda, koniec)."""
    dy, dx = RUCHY[akcja]
    y, x = stan[0] + dy, stan[1] + dx
    if not (0 <= y < ROZMIAR and 0 <= x < ROZMIAR):
        return stan, -5, False  # ściana — stoi w miejscu
    if (y, x) in PRZESZKODY:
        return (y, x), -20, True  # przeszkoda — koniec epizodu
    if (y, x) == CEL:
        return (y, x), 50, True  # cel!
    return (y, x), -1, False  # każdy krok kosztuje — zachęta do skrótów


# ── Q-learning: tablica [y][x][akcja] z oceną "ile mi się opłaca ten ruch" ──
Q = np.zeros((ROZMIAR, ROZMIAR, 4))
ALFA, GAMMA, EPIZODY = 0.1, 0.95, 600

historia_nagrod, historia_krokow, historia_finalow = [], [], []
for epizod in range(EPIZODY):
    # epsilon maleje: na początku sama eksploracja, potem coraz więcej wiedzy
    epsilon = max(0.05, 1.0 - epizod / (EPIZODY * 0.6))
    stan, suma, kroki, koniec = START, 0, 0, False
    while not koniec and kroki < 200:
        akcja = rng.integers(4) if rng.random() < epsilon else int(np.argmax(Q[stan]))
        nowy, nagroda, koniec = krok(stan, akcja)
        # sedno Q-learningu: podciągnij ocenę ruchu w stronę tego, co faktycznie wyszło
        Q[stan][akcja] += ALFA * (
            nagroda + GAMMA * np.max(Q[nowy]) * (not koniec) - Q[stan][akcja]
        )
        stan, suma, kroki = nowy, suma + nagroda, kroki + 1
    historia_nagrod.append(suma)
    historia_krokow.append(kroki)
    historia_finalow.append(
        "cel" if stan == CEL else ("przeszkoda" if koniec else "limit")
    )


# ── Krzywa uczenia ──
def wygladz(dane, okno=25):
    return np.convolve(dane, np.ones(okno) / okno, mode="valid")


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(historia_nagrod, alpha=0.25, color="#1f77b4")
ax1.plot(range(24, EPIZODY), wygladz(historia_nagrod), lw=2.5, color="#1f77b4")
ax1.axhline(0, color="grey", ls="--", lw=1)
ax1.set_xlabel("Epizod")
ax1.set_ylabel("Suma nagród")
ax1.set_title(
    "Nagroda rośnie — agent przestaje wpadać na przeszkody", fontweight="bold"
)
ax2.plot(historia_krokow, alpha=0.25, color="#d62728")
ax2.plot(range(24, EPIZODY), wygladz(historia_krokow), lw=2.5, color="#d62728")
ax2.axhline(14, color="green", ls="--", lw=1.5, label="trasa optymalna: 14 kroków")
ax2.set_xlabel("Epizod")
ax2.set_ylabel("Liczba kroków")
ax2.legend()
ax2.set_title("Kroków ubywa — agent znajduje skrót", fontweight="bold")
plt.tight_layout()
plt.show()


# ── Porównanie: agent losowy kontra nauczony ──
def przejdz(losowo):
    stan, sciezka, koniec, kroki = START, [START], False, 0
    while not koniec and kroki < 200:
        akcja = rng.integers(4) if losowo else int(np.argmax(Q[stan]))
        stan, _, koniec = krok(stan, akcja)
        sciezka.append(stan)
        kroki += 1
    return sciezka, stan == CEL


sukcesy_losowe = sum(przejdz(True)[1] for _ in range(100))
sciezka, dotarl = przejdz(False)
print(f"🎲 Agent losowy:   dotarł do celu w {sukcesy_losowe}/100 prób")
print(
    f"🧠 Agent nauczony: {'dotarł' if dotarl else 'NIE dotarł'} do celu w {len(sciezka)-1} krokach"
)


def podsumuj(etykieta, wycinek):
    fin = historia_finalow[wycinek]
    cel = fin.count("cel")
    print(
        f"{etykieta}  cel: {cel:>2}/{len(fin)}   przeszkoda: {fin.count('przeszkoda'):>2}/{len(fin)}"
        f"   śr. nagroda: {np.mean(historia_nagrod[wycinek]):>6.0f}"
    )


print("\n── Jak kończyły się epizody ──")
podsumuj("📉 pierwsze 50:", slice(None, 50))
podsumuj("📈 ostatnie  50:", slice(-50, None))
print("\n💡 Uwaga: na początku epizody są KRÓTKIE nie dlatego, że agent jest sprawny,")
print(
    "   tylko dlatego, że szybko rozbija się o przeszkodę. Patrz na nagrodę, nie na kroki."
)

# ── Mapa: co agent uważa za najlepszy ruch w każdym polu ──
strzalki = {0: "↑", 1: "↓", 2: "←", 3: "→"}
print("\n🗺️  Wyuczona strategia (▓ przeszkoda, ★ cel):")
for y in range(ROZMIAR):
    wiersz = ""
    for x in range(ROZMIAR):
        if (y, x) in PRZESZKODY:
            wiersz += " ▓"
        elif (y, x) == CEL:
            wiersz += " ★"
        elif Q[y, x].any():
            wiersz += " " + strzalki[int(np.argmax(Q[y, x]))]
        else:
            wiersz += " ·"
    print("   " + wiersz)

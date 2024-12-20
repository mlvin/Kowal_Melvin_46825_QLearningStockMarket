# Kowal_Melvin_46825_QLearningStockMarket
Ein praktisches Reinforcement Learning Projekt zur Vorlesung "Einführung Artificial Intelligence" im Wintersemester 2024 an der LFH.
### Installation
1. GitHub-Repository klonen:
   ```bash
   git clone https://github.com/mlvin/Kowal_Melvin_46825_QLearningStockMarket.git
2. In den Ordner des Repositories navigieren:
   ```bash
   cd Kowal_Melvin_46825_QLearningStockMarket
3. Erforderliche Pakete installieren:
   ```bash
   pip install -r requirements.txt
4. Algorithmus ausführen:
   ```bash
   python3 main.py
\
\
Bei Installationsproblemen gern auf mich zukommen. :)
   
# Beantwortung der Fragen zum Projekt
## Was ist Q-Learning?
Q-Learning ist eine spezielle Form des Reinforcement Learnings. Bei Reinforcment Learning handelt es sich um einen Ansatz, bei dem der Agent keine vordefinierten Antworten oder Anweisungen bekommt, sondern durch Trial und Error dazulernt. Der Agent bekommt hierbei von seiner Umgebung (Environment) Informationen bereitgestellt und erhält darüber hinaus Feedback in Form von Belohnung oder Bestrafung durch eine Reward-Funktion. Ziel des Agenten ist es, die Belohnung zu maximieren und somit das vorgegebene Problem optimal zu lösen (siehe Abbildung).\
\
![image](https://github.com/user-attachments/assets/d42eab39-201e-412b-a8d1-f3f2941d4474)\
*"The agent–environment interaction in reinforcement learning", Quelle: Sutton, R./Barto, A. (2018), S.54.*
\
\
Q-Learning ist modellfrei, sprich der Agent benötigt keine explizite Modellvorstellung seiner Umgebung, sondern erlent die Policy (die Strategie des Agenten das Problem zu lösen) über das Bestimmen sogenannter Q-Werte. Q-Werte repräsentieren die "Qualität" einer Aktion zu einem Zustand (Sate-Action-Pair) und sind für den Agenten folglich Entscheidungsgrundlage dafür, welche Aktion in einem bestimmten State zu wählen ist. Q-Werte werden für das jeweilige State-Action-Pair im Lernprozess des Agenten durch die sogenannte Bellman-Gleichung aktualisiert:

$Q_{\text{new}}(S_t, A_t) \leftarrow (1 - \alpha) \cdot Q(S_t, A_t) + \alpha \cdot \left( R_{t+1} + \gamma \cdot \max_a Q(S_{t+1}, a) \right)$\
\
$(1 - \alpha) \cdot Q(S_t, A_t)$ stellt die Gewichtung des alten Q-Wertes dar, wobei α die Lernrate ist. Folglich gilt: Je höher die Lernrate, desto stärker werden neue Informationen stärker gewichtet als alte.\
\
$R_{t+1}$ stellt die Belohnung aus der Reward-Funktion für die getroffene Entscheidung dar.\
\
$\gamma \cdot \max_a Q(S_{t+1}, a)$ wobei γ der Discount-Faktor ist, sprich wie stark zukünftige Belohnungen gewichtet werden. Ist der Wert beispielsweise nahe 1, so sind zukünftige Belohnungen fast genauso wichtig, wie die unmittelbare Belohnung. Der zweite Teil stellt den maximalen Q-Wert für den nächsten Zustand dar, also die bestmöglich gefundene Entscheidung (im Falle des Trading Environments: Kaufen, oder Verkaufen in einem State).\
\
Der Q-Table (Q-Tabelle) stellt die zentrale Datenstruktur des Q-Learnings dar. In dieser Werten alle State-Action-Pairs sowie ihrer Zugehörigen Q-Werte gespeichert. Wie aus dem Namen hervorgeht handelt es sich hierbei um eine tabellen- bzw. Matrixstruktur, welche während des Lernprozesses des Agenten, basierend auf obiger Bellmann-Gleichung, dynamisch angepasst wird.\
\
Eine zentraler Aspekt des Q-Learnings ist darüber hinaus die Trennung in die sogenannte Exploration und Exploitation Phase: Da die Q-Tabelle zu Beginn nicht (oder mit zufälligen Werten) befüllt ist, muss der Agent zunächst seine Umgebung (Environment) kennenlernen. Dies geschieht in der Exploration Phase, in welcher der Agent zufällig Aktionen ausführt und entsprechend seinen Q-Table aktualisiert. In der Exploitation Phase nutzt der Agent widerum seine gesammelten Erkenntnisse aus und wendet seine optimal gefundene Policy an: Der Agent wählt für den jeweiligen State die beste Action aus den State-Action-Pairs aus (höchster Q-Wert) und durchläuft so das Problem.\
\
Für die Trennung zwischen Exploration und Exploitation Phase gibt es mehrer Ansätze. Ein populärer und verwendeter Ansatz in diesem Projekt, ist die Epsilon-Greedy-Strategie. Mit Beginn des Lernens hat das Epsilon zunächst einen Wert von 1, sodass der Agent exploriert. Im Verlauf des Trainings wird der Wert von Epsilon jedoch zunehmend reduziert. Dies erfolgt durch den sogenannten Epsilon-Decay, welcher den schrittweisen Abfall von Epsilon bei jeder Iteration beschreibt. Dadurch nimmt die Wahrscheinlichkeit für Exploration mit der Zeit ab, während die Exploitation (Nutzung des bereits erlernten Wissens) zunehmend bevorzugt wird und der Agent somit das Problem mit seiner erlernten Policy löst.

## Wie ist der Agent aufgebaut?
Der Agent funktioniert nach obig erklärten Prinzip und setzt sich aus einer gleichnamigen Klasse "Agent" zusammen. Die Klasse Agent erwartet folgende Übergabeparameter im Konstruktor:
```python
class Agent:
  def __init__(self, dataframe=None, n_epochs=20000, frame_bound=(5, 30), learning_rate=0.7, discount_factor=0.5,
                   epsilon=1, epsilon_min=0.01,
                   epsilon_decay=0.9996):
```
Dem Agenten kann ein Dataframe übergeben werden, um eigene Aktiencharts zu verwenden, ist kein Dataframe angegeben, so wird der Kurs der Google Aktie verwendet, welche standardmäßig im Environment hinterlegt ist. Der Frame Bound definiert den Bereich auf der Chart auf dem der Agent trainiert wird. Die anderen Attribute ergeben sich aus der obigen Erkärung.\
Zur Modellierung der Umwelt wird auf die Klasse "StocksEnv" aus dem GitHub-Repository "gym-anytrading" (https://github.com/AminHP/gym-anytrading) zugegriffen, dass ein Aktien-Environment zur Verfügung stellt und auf dem OpenAI-Gym basiert. Dieses Environemnt wird an ein eigens-entwickeltes Environemnt "CustomTradingEnv" vererbt, welches wesentliche Methoden wie die Reward-Funktion, oder States implementiert (siehe Frage: Wie wird die Umwelt modelliert? Sowie dem folgenden Klassendiagramm).
#### Klassendiagramm des Projektes
![image](https://github.com/user-attachments/assets/2506f3fd-4978-4d9c-9319-066808cdb03a)
*Das StocksEnv wird nur mit den relevanten und unbedingt erforderlichen Attributen dargestellt, da es sich um eine umfangreiche Klasse handelt*
\
\
Der Methode "train_agent" trainiert den Agenten mithilfe der übergebenen Attribute. Dabei stehen dem Agenten die Aktionen Buy (Kaufen) und Sell (Verkaufen / Short) zur Verfügung. In jeder Iteration berechnet der Agent seinen nächsten Schritt, indem er die aktuelle Q-Tabelle nutzt und sie mithilfe der Bellman-Gleichung aktualisiert. Durch den Epsilon-Greedy-Ansatz entscheidet der Agent, ob er auf Basis seines bisherigen Wissens handelt (Exploitation) oder neue Aktionen ausprobiert (Exploration). Der Lernprozess wird über die Anzahl der definierten Epochs durchgeführt. Anschließend wird am Ende das Ergebnis der letzten Iteration auf der Chart dargestellt und der Q-Table ausgegeben. Auf den genauen Aufbau des Q-Tables wird bei der Fragestellung "Wie wird der Reward repräsentiert?" eingegangen.\

## Wie wird die Umwelt modelliert?
Wie obig erläutert, wird die Umwelt durch ein eigenes Environment modelliert, welches von der Klasse "StocksEnv" aus dem Repository "gym-anytrading" erbt. Grund hierfür ist, dass der Reward sowie State-Representation des gegebenen Environments simpel ausgestaltet sind und der State lediglich durch den Schlusspreis sowie der Preisdifferenz zum Vortag dargestellt. Der Reward bezieht sich gleichermaßen lediglich darauf ob die Aktion des Agenten gewinnbringend oder verlustreich ist. Dies ist für den gewählten Ansatz eines Reinforcement Agenten der Q-Learning einsetzt nicht tragbar, da der Agent so keine fundierten Entscheidungen treffen kann, wie sich auch durch Tests gezeigt hat.\
\
Das CustomTradingEnv implementiert den State (_get_observation()) so, dass der State aus einem Tupel aus den Indikatoren: Simple Moving Average (SMA), Relative Strength Index (RSI), Momentum sowie Volatilität und Preisänderung zum Vortag, besteht und zurückgegeben wird. SMA, RSI, Momentum und Preisänderung können hierbei die Werte 1 -> Buy oder 0 -> Sell annehmen, da der State direkt "Handlungsempfehlungen" ableiteitet um die Dimension des Q-Tables zu schonen, während es sich bei der Volatilität um einen diskreten Wert handelt. Diese Indikatoren sind gängig im Bereich des Aktienhandels und zeigen in der Kombination die besten Ergebnisse.
\
**Probleme mit diskreten Werten:**\
\
Das Nutzen von diskreten Werten kann zu Problemen führen, da die Anzahl an States somit praktisch unbegrenzt ist wodurch sich die Performance über große Zeiträume stark verschlechtert. Eine mögliche Lösung hierfür wäre die Quantisierung, bei der kontinuierliche Werte, wie z.B. Preisänderungen, auf einen festen Bereich von 0 bis 1 komprimiert und in gleichmäßige Schritte (z.B. in 0.1er-Schritten) unterteilt werden. Alternativ können auch Handlungsempfehlungen wie "Buy" oder "Sell" abgeleitet werden. Hinsichtlich der Volatilität hat dies die Genaugikeit des Agenten jedoch stark negativ beeinflusst, sodass die Volatilität als diskreter Wert belassen wurde, insbesondere da Performanceeinbußen bei den auferlegten Zeiträumen nicht wahrnehmbar waren.\
\
Darüber hinaus überschreibt das "CustomEnv" die Methode "_update_profit()", da in der Oberklasse "StocksEnv" Transaktionsgebühren mit in der Ermittlung des Profits einfließen. Zur Komplexitätsreduzierung wurde diese Eigenschaft in der Methode des "CustomEnv" entfernt.

## Wie wird der Reward representiert?
Der Reward bezieht sich auf die im State ermittelten Indikatoren um den Agenten bestmöglich im Lernprozess zu unterstützen und ist ebenfalls als Methode im "CustomTradingEnv" implementiert. Datentyptechnisch handelt es sich um einen Float, folglich kann der Reward positiv als auch negativ sein.\
\
Die Methode fragt mehrere Konstellationen ab und errechnet daraus den zurückzugebenden Reward. Für die Aktion "Buy" gilt beispielsweise (für "Sell" gilt ähnliches nur in anderer Richtung):
```python
if self._position == Positions.Short:
    # Profit aus der Schließung einer Short-Position
    step_reward += (last_trade_price - current_price) * 3
elif self._position == Positions.Long:
    # Bestrafung für redundante Kaufaktionen, wenn bereits in einer Long-Position
    step_reward -= 0.5

# Belohnung für Kauf in einem überverkauften Markt (RSI < 30)
if rsi_indicator == 1:  # RSI < 30: Überverkauft
    step_reward += 1.5  # Höhere Belohnung für Kauf bei günstigen Marktbedingungen

# Bestrafung für Kauf in einem Abwärtstrend (momentum < 0)
if momentum == 0:  # Kein Momentum
    step_reward -= 1  # Kleine Bestrafung für Kauf ohne Momentum
```
Darüber hinaus wird der Reward in Abhängigkeit der größe des Verlustes / Profites skaliert und langsames Handeln bestraft:
```python
if step_reward < 0:
    # Verstärkte Strafe für größere Verluste, abhängig vom Preisunterschied
    step_reward *= 1 + (abs(price_diff) / current_price) * 0.7
elif step_reward > 0:
    # Verstärkte Belohnung für größere Gewinne, abhängig vom Preisunterschied
    step_reward *= 1.5 + (abs(price_diff) / current_price) * 0.3

# Zusätzliche Strafe für langsames Handeln
if abs(price_diff) < 0.02 * current_price:  # Wenn der Preisunterschied zu gering ist
    step_reward -= 0.5
```
## Vorstellung der Ergebnisse eines Durchlaufes
Folgender Durchlauf wurde mit den Standardparametern des Agenten durchgeführt. Der genutzte Aktienkurs ist der Google-Aktienkurs, welcher nativ vom Environment zur Verfügung gestellt wird.
Der Finale Durchlauf ergab folgendes Ergebnis:\
\
![Figure_1](https://github.com/user-attachments/assets/50578806-cd31-419b-bb44-2f41f27a4867)
\
\
**Ausschnitt aus dem Q-Table:**

| State                                | Q-values (Buy, Sell)                   |
|--------------------------------------|--------------------------------|
| (1, -1, 1, np.float32(4.03), 0)      | [-6.2999252   0.67106444]     |
| (1, -1, 1, np.float32(3.35), 1)      | [-6.79130352  3.3500067 ]     |
| (0, -1, 0, np.float32(2.06), 1)      | [10.74121976  5.09576441]     |
| (1, -1, 0, np.float32(1.99), 0)      | [ 7.60168231 14.1954875 ]     |
| (1, -1, 0, np.float32(1.73), 1)      | [20.71418571 -0.50708961]     |

Es zeigt sich, dass der Agent einen Profit von 1.12 erwirtschaften konnte (12% Rendite), wobei der maximal mögliche Gewinn bei 1.38 (38%) liegt.
So ist der Agent profitabel, trifft jedoch nicht an jedem Punkt die bestmögliche Entscheidung. Dies ist darauf zurückzuführen, dass Aktienkurse gemeinhin als sehr komplex und stochastisch beschrieben werden können. Die zugrunde liegenden Marktmechanismen unterliegen zahlreichen, oft unvorhersehbaren Faktoren wie Wirtschaftsdaten, geopolitischen Ereignissen und Marktstimmungen, die eine präzise Vorhersage anhand weniger Indikatoren im State erschweren. Nichtdestotrotz lässt sich das Ergebnis als gut klassifizieren, da eine Rendite von 12% an 50 Hanadelstagen ein akzeptables Ergebnis darstellt und sich am Verhalten des Agenten zeigen lässt, dass dieser begründbare Entscheidungen trifft. So ist in obiger Abbildung zu sehen, dass der Agent dazu neigt sich für Buy zu entscheiden, wenn die Kurse niedrig sind und in Hochphasen dazu tendiert zu verkaufen. Dies wird insbesondere ab Handelstag 35 deutlich.

## Die Besonderheiten des Aktienmarktes, wieso interessant als Problem?
Wie bereits erläutert lässt sich der Aktienmarkt als stochastisch beschreiben. Es gibt eine Vielzahl von Ereignissen, welche die Grundlage für Kursbewegungen darstellen können. Dies macht den Aktienmarkt besonders interessant für Reinforcement Learning, da Ansätze wie Q-Learning den Markt als deterministisch betrachten, obwohl dieser stark von Zufallsfaktoren geprägt ist, sodass nur näherungsweise "optimale" Ergebnisse erzielt werden können, anders als in "typischen" Reinforcement Learning Problemen, wie dem Entkommen eines Labyrinths. Beim Q-Learning Ansatz bieten sich darüber hinaus viele verschiedene Ausgestaltungen für die Darstellung des Rewards sowie der States.

# Erktennisse und Probleme bei der Implementierung
Die Modellierung der Umgebung, insbesondere der States und der Reward-Funktion, hat sich als eine erhebliche Herausforderung herausgestellt, was vor allem auf die beschriebenen Eigenheiten des Aktienmarktes zurückzuführen ist. Diese Komplexität erschwert es dem Agenten, eine konsistent profitable Handelsstrategie zu entwickeln. Ein Beispiel hierfür ist die Schwierigkeit, eine Reward-Funktion zu gestalten, die den Agenten dazu anregt, den Gewinn langfristig zu maximieren, anstatt sich auf die Minimierung von Verlusten zu konzentrieren. Dies führt dazu, dass der Agent in bestimmten Fällen zu konservativ agiert und potenziell gewinnbringende Gelegenheiten übersieht.\
\
Ein weiteres Problem bestand in der Auswahl und Kombination geeigneter Indikatoren für die Zustandsdarstellung (State). Der Markt ist von Natur aus dynamisch und kann unterschiedliche Phasen wie hohe Volatilität oder fallende Kurse durchlaufen. Es war herausfordernd, eine Set von Indikatoren zu finden, das in verschiedenen Marktumfeldern funktioniert und gleichzeitig profitabel bleibt.

# Grundlegend verwendete Literatur
Geron, A. (2023): Praxiseinstieg Machine Learning - Konzepte, Tools und Techniken für intelligente Systeme, Heidelberg 2023.\
\
Sutton, R./Barto, A. (2018): Reinforcement Learning: An Introduction, Stanford 2018.\
\
Frattini, A./Bianchini, I./Garzonio, A. (2022): Financial Technical Indicator and Algorithmic Trading Strategy Based on Machine Learning and Alternative Data, in: Risks Vol. 10.\
\
deeplizard (2018): Q-Learning Explained - A Reinforcement Learning Technique, https://www.youtube.com/watch?v=qhRNvCVVJaA.





# Kowal_Melvin_46825_QLearningStockMarket
Praktisches Reinforcement Learning Projekt zur Vorlesung "Einführung Artificial Intelligence" im Wintersemester 2024 an der LFH.

# Beantwortung der Fragen zum Projekt
**Was ist Q-Learning?**\
Q-Learning ist eine spezielle Form des Reinforcement Learnings. Bei Reinforcment Learning handelt es sich um einen Ansatz, bei dem der Agent keine vordefinierten Antworten oder Anweisungen bekommt, sondern durch Trial und Error dazulernt. Der Agent bekommt hierbei von seiner Umgebung (Environment) Informationen bereitgestellt und erhält darüber hinaus Feedback in Form von Belohnung oder Bestrafung durch eine Reward-Funktion. Ziel des Agenten ist es, die Belohnung zu maximieren und somit das vorgegebene Problem optimal zu lösen (siehe Abbildung).\
\
![image](https://github.com/user-attachments/assets/d42eab39-201e-412b-a8d1-f3f2941d4474)

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
Eine zentraler Aspekt des Q-Learnings ist darüber hinaus die Trennung in die sogenannte Exploration und Exploitation Phase: Da die Q-Tabelle zu Beginn nicht (oder mit zufälligen Werten) befüllt ist, muss der Agent zunächst seine Umgebung (Environment) kennenlernen. Dies geschieht in der Exploration Phase, in welcher der Agent zufällig Aktionen ausführt und entsprechend seinen Q-Table aktualisiert. In der Exploitation Phase nutzt der Agent widerum seine gesammelten Erkenntnisse aus und wendet seine optimal gefundene Policy an: Der Agent wählt für den jeweiligen State das beste State-Action-Pair aus (höchster Q-Wert) und durchläuft so das Problem.\
\
Für die Trennung zwischen Exploration und Exploitation Phase gibt es mehrer Ansätze. Ein populärer und verwendeter Ansatz in diesem Projekt, ist die Epsilon-Greedy-Strategie. Mit Beginn des Lernens hat das Epsilon zunächst einen Wert von 1, sodass der Agent exploriert. Ein weiterer Faktor, das sogenannte Epsilon-Decay stellt den Abfall des Epsilons für jede Iteration dar. Folglich sind 




**Wie ist der Agent aufgebaut?**
Der Agent folg dem Q-Learning Ansatz, bei dem 

'Aufbau Agenten, Wie wird Reward representiert/Ausgegeben, Vorstellung Q-Learning, wie wird die Umwelt modelliert? Vorstellung des Ergebnisses eines Durchlaufes, Besonderheiten des Aktienmarktes/Warum interessant als Problem?

![image](https://github.com/user-attachments/assets/2506f3fd-4978-4d9c-9319-066808cdb03a)


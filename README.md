# Kowal_Melvin_46825_QLearningStockMarket
Praktisches Reinforcement Learning Projekt zur Vorlesung "Einführung Artificial Intelligence" im Wintersemester 2024 an der LFH.

# Beantwortung der Fragen zum Projekt
**Was ist Q-Learning?**\
Q-Learning ist eine spezielle Form des Reinforcement Learnings. Bei Reinforcment Learning handelt es sich um einen Ansatz, bei dem der Agent keine vordefinierten Antworten oder Anweisungen bekommt, sondern durch Trial und Error dazulernt. Der Agent bekommt hierbei von seiner Umgebung (Environment) Informationen bereitgestellt und erhält darüber hinaus Feedback in Form von Belohnung oder Bestrafung durch eine Reward-Funktion. Ziel des Agenten ist es, die Belohnung zu maximieren und somit das vorgegebene Problem optimal zu lösen (siehe Abbildung).\
\
![image](https://github.com/user-attachments/assets/d42eab39-201e-412b-a8d1-f3f2941d4474)

\
Q-Learning ist modellfrei, sprich der Agent benötigt keine explizite Modellvorstellung seiner Umgebung, sondern erlent die Policy (die Strategie des Agenten das Problem zu lösen) über das bestimmen sogenannter Q-Werte. Q-Werte repräsentieren die "Qualität" einer Aktion zu einem Zustand (Sate-Action-Pair) und sind für den Agenten folglich Entscheidungsgrundlage, welche Aktion in einem bestimmten State zu wählen ist. Q-Werte werden im Lernprozess des Agenten durch die sogenannte Bellman-Gleichung aktualisiert


**Wie ist der Agent aufgebaut?**
Der Agent folg dem Q-Learning Ansatz, bei dem 

'Aufbau Agenten, Wie wird Reward representiert/Ausgegeben, Vorstellung Q-Learning, wie wird die Umwelt modelliert? Vorstellung des Ergebnisses eines Durchlaufes, Besonderheiten des Aktienmarktes/Warum interessant als Problem?

![image](https://github.com/user-attachments/assets/2506f3fd-4978-4d9c-9319-066808cdb03a)


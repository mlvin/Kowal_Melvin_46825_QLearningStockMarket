import numpy as np
import matplotlib.pyplot as plt
from gym_anytrading.datasets import STOCKS_GOOGL
from gym_anytrading.envs import StocksEnv, Actions, Positions


class CustomTradingEnv(StocksEnv):
    # Konstruktor inkl. Vererbung des StocksEnv's
    """
        Angepasstes Aktien-Envrionment hinsichtlich State (observation), Reward sowie Handelsgebühren, welches von dem "StockEnv" aus dem Github-Repository gym-anytrading erbt.

        Attributes:
            df: DataFrame -> Aktienchart
            window_size: int -> Untere Grenze an dem der Agent auf der Chart "startet" -> Fenstergröße.
            frame_bound: tuple -> Tupel, welches den Zeitraum auf der Aktienchart definiert

        """
    def __init__(self, df, window_size, frame_bound):
        super().__init__(df=df, window_size=window_size, frame_bound=frame_bound)

    # Reward-Funktion des Environments, welche sich auf SMA, RSI & Momentum bezieht.
    def _calculate_reward(self, action):
        """
        Berechnet den Reward für die gewählte Aktion des Agenten.

        Attributes:
            action: boolean -> die gewählte Aktion des Agenten (Buy=1, Sell=0)

        Returns:
            step_reward: float -> Der ermittelte Reward
        """
        # Initiale Belohnung auf 0 setzen
        step_reward = 0
        current_price = self.prices[self._current_tick]
        last_trade_price = self.prices[self._last_trade_tick]
        price_diff = current_price - last_trade_price

        # Ausgabe der Marktindikatoren (RSI, SMA und Momentum) aus dem State -> Volatilität zur Vereinfachung nicht in der Rewardfunktion
        sma_indicator, rsi_indicator, momentum, _, _ = self._get_observation()

        # Falls sich der Agent für Buy (1) entschieden hat:
        if action == Actions.Buy.value:
            if self._position == Positions.Short:
                # Profit aus der Schließung einer Short-Position
                step_reward += (last_trade_price - current_price) * 3
            elif self._position == Positions.Long:
                # Bestrafung für redundante Kaufaktionen, wenn bereits in einer Long-Position
                step_reward -= 0.5

            # Belohnung für Kauf in einem überverkauften Markt (RSI < 30)
            if rsi_indicator == 1:  # RSI < 30: Überverkauft
                step_reward += (
                    1.5  # Höhere Belohnung für Kauf bei günstigen Marktbedingungen
                )

            # Bestrafung für Kauf in einem Abwärtstrend (momentum < 0)
            if momentum == 0:  # Kein Momentum
                step_reward -= 1  # Kleine Bestrafung für Kauf ohne Momentum

        # Falls sich der Agent für Sell (0) entschieden hat:
        elif action == Actions.Sell.value:
            if self._position == Positions.Long:
                # Profit aus der Schließung einer Long-Position
                step_reward += (current_price - last_trade_price) * 3
            elif self._position == Positions.Short:
                # Bestrafung für redundante Verkaufsaktionen, wenn bereits in einer Short-Position
                step_reward -= 0.5

            # Belohnung für Verkauf in einem überkauften Markt (RSI > 70)
            if rsi_indicator == -1:  # RSI > 70: Überkauft
                step_reward += (
                    1.5  # Höhere Belohnung für Verkauf bei ungünstigen Marktbedingungen
                )

            # Bestrafung für Verkauf im Aufwärtstrend (momentum > 0)
            if momentum == 1:  # Positive Momentum
                step_reward -= 1.5  # Bestrafung für Verkauf bei positivem Momentum

        # Verstärkter Fokus auf Preisänderung und Momentum
        if step_reward < 0:
            # Verstärkte Strafe für größere Verluste, abhängig vom Preisunterschied
            step_reward *= 1 + (abs(price_diff) / current_price) * 0.7
        elif step_reward > 0:
            # Verstärkte Belohnung für größere Gewinne, abhängig vom Preisunterschied
            step_reward *= 1.5 + (abs(price_diff) / current_price) * 0.3

        # Zusätzliche Strafe für langsames Handeln
        if (
            abs(price_diff) < 0.02 * current_price
        ):  # Wenn der Preisunterschied zu gering ist
            step_reward -= 0.5

        # Rückgabe des Rewards
        return step_reward

    def _update_profit(self, action):
        """
        Überschreibung der Profit-Methode um Handelsgebühren zu ignorieren.

        Attributes:
            action: boolean -> die gewählte Aktion des Agenten (Buy=1, Sell=0)

        Returns:
            none (Agent greift über das Environment auf den Profit zu)
        """
        trade = False
        if (action == Actions.Buy.value and self._position == Positions.Short) or (
            action == Actions.Sell.value and self._position == Positions.Long
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position == Positions.Long:
                shares = self._total_profit / last_trade_price
                self._total_profit = shares * current_price

    def _get_observation(self):
        """
        Wandelt den State (hier observation) des environments in eine benutzerdefinierte Darstellung mit zusätzlichen Rückggabeparametern
        um, damit präzisere Vorhersagen erstellt werden können. Die Standard-observation des environments gibt lediglich
        den aktuellen Kurs sowie Veränderung zum Vortag an. -> Nicht geeignet für Q-Learning

        Returns:
            tuple: Ein Tupel, dass die benutzerdefinierte observation darstellt.
            -> Indikatoren für SMA, RSI & Momentum: 0 -> Sell, 1 -> Buy
            Volatilität, Preisänderung -> Diskrete Werte
        """

        # Ursprüngliche observation aus der Library
        observation = self.signal_features[
            (self._current_tick - self.window_size + 1) : self._current_tick + 1
        ]

        # Alle Schlusskurse der vergangenen Tage laden -> definiert durch den unteren Frame-Bound
        prices = [obs[0] for obs in observation]

        # Berechnung des kurzfristigen Gleitenden Durchschnitts (SMA) für den Zeitraum
        sma = np.mean(prices)  # SMA ist der Mittelwert der letzten 'window_size' Tage
        sma_indicator = (
            1 if observation[-1][0] > sma else 0
        )  # Trend: Wenn der letzte Kurs über dem SMA liegt -> Long

        # Berechnung des Relative Strength Index (RSI) für den Markt
        price_changes = [
            obs[1] for obs in observation
        ]  # Preisänderungen der letzten X Tage

        avg_positive = (
            np.mean([change for change in price_changes if change > 0])
            if any(change > 0 for change in price_changes)
            else 0
        )
        avg_negative = (
            np.mean([-change for change in price_changes if change < 0])
            if any(change < 0 for change in price_changes)
            else 1e-6 # um Divison by Zero zu verhindern
        )

        rs = avg_positive / avg_negative
        rsi = 100 - (100 / (1 + rs))

        # RSI Indikator: Überkauft (RSI > 70) -> Sell, Überverkauft (RSI < 30) -> Buy, Neutral (30 < RSI < 70) -> Neutral
        rsi_indicator = 1 if rsi < 30 else (-1 if rsi > 70 else 0)

        # Berechnung des Momentums: Ist der Preis gestiegen oder gefallen im Vergleich zum ersten Tag der Beobachtungsperiode?
        # Wenn ja Buy (1), sonst Sell (0)
        momentum = 1 if (observation[-1][0] - observation[0][0]) > 0 else 0

        # Berechnung der Volatilität (Standardabweichung der letzten Preisänderungen)
        price_volatility = round(
            np.std(prices), 2
        )  # Standardabweichung der Preisänderungen als Maß für die Volatilität

        # Berechnung der Preisänderung im Vergleich zum vorherigen Tag
        price_change = (
            1 if price_changes[0] - price_changes[-1] > 0 else 0
        )  # Die Veränderung des Preises zum letzten Handelstag

        # Rückgabe der Indikatoren als Tupel
        return (sma_indicator, rsi_indicator, momentum, price_volatility, price_change)



class Agent:
    """
    Der Agent verwendet Q-Learning zur Maximierung seines Profits im Rahmen einer historischen Makrtchart.
    Hierbei kann der Agent zwischen kaufen (Buy) oder verkaufen (sell) basierend auf den Marktbedinungen, welche durch
    die CustomTradingEnv Klasse bereitgestellt werden, entscheiden. Diese stellt durch den State (_get_observation) sowie
    der Reward-Funktion, die Marktdaten und Indikatoren zur Entscheidungsfindung zur Verfügung. Das Ergebnis der letzten
    Iteration wird als Chart ausgegeben und der Q-Table in der Konsole ausgegeben.

    Attributes:
           df: DataFrame, das die Aktienkursdaten enthält.
           n_epochs: Anzahl der Durchläufe für das Training des Agenten.
           frame_bound: Tupel, welches den Zeitraum auf der Aktienchart definiert -> Wichtig: Der erste Eintrag
            muss >= 1 sein, da der Agent die Daten mindestens eines Tages als Historie benötigt. Je höher der untere
            Frame-Bound, mit desto mehr Historie startet der Agent. -> frame_bound = (10, 40): 10 Tage als Historie
            für das Training.
           learning_rate: Die Lernrate zur Aktualisierung der Q-Tabelle.
           discount_factor: Der Discount-Factor, um zukünftige Belohnungen zu berücksichtigen.
           epsilon: Die Explorationsrate für den epsilon-greedy Ansatz.
           epsilon_min: Der minimale Wert für epsilon.
           epsilon_decay: Der Abklingfaktor für epsilon nach jedem Epoch.
           
    """

    def __init__(
        self,
        dataframe=None,
        n_epochs=20000,
        frame_bound=(5, 50),
        learning_rate=0.7,
        discount_factor=0.5,
        epsilon=1,
        epsilon_min=0.01,
        epsilon_decay=0.9996,
    ):

        self.df = dataframe
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.frame_bound = frame_bound
        self.window_size = self.frame_bound[0]

        # Erstellung eines leeren Q-Tables -> Die Q-Tabelle wird dynamisch befüllt, um die Performance zu verbessern.
        self.q_table = {}

        # Falls kein Data-Frame angegeben, wird auf die Google-Aktie aus dem Environment zugegriffen.
        if self.df is None:
            self.df = STOCKS_GOOGL

        #Environment erstellen
        self.env = CustomTradingEnv(
            df=self.df, window_size=self.window_size, frame_bound=self.frame_bound
        )

        # Anzahl an Auswahlmöglichkeiten -> 2, da Buy & Sell
        self.n_actions = self.env.action_space.n

    def train_agent(self):
        """
        Methode zum Trainieren des Agenten. -- Die letzte Iteration wird hierbei als Chart ausgegeben
                Attributes:
                   df: DataFrame, das die Aktienkursdaten enthält.
                   n_epochs: Anzahl der Durchläufe für das Training des Agenten.
                   frame_bound: Tupel, welches den Zeitraum auf der Aktienchart definiert -> Wichtig: Der erste Eintrag
                    muss >= 1 sein, da der Agent die Daten mindestens eines Tages als Historie benötigt. Je höher der untere
                    Frame-Bound, mit desto mehr Historie startet der Agent. -> frame_bound = (10, 40): 10 Tage als Historie
                    für das Training.
                   learning_rate: Die Lernrate zur Aktualisierung der Q-Tabelle.
                   discount_factor: Der Discountfactor, um zukünftige Belohnungen zu berücksichtigen.
                   epsilon: Die Explorationsrate für den epsilon-greedy Ansatz.
                   epsilon_min: Der minimale Wert für epsilon.
                   epsilon_decay: Der Abklingfaktor für epsilon nach jedem Epoch.
        """
        # Training für die Anzahl an festgelegten Epochs
        for epoch in range(self.n_epochs):
            # Initialen state des Environments abrufen
            observation, _ = self.env.reset()
            state = observation
            done = False
            total_reward = 0

            # Iteration über jeden Schlusskurs
            while not done:
                # Prüfung ob der State-Tuple aus der observation bereits im Q-Table vorhanden ist. -> Falls nein hinzufügen und mit dem Wert 0 belegen.
                if state not in self.q_table:
                    self.q_table[state] = np.zeros(self.n_actions)

                # Entscheidung ob Exploration oder Exploitation mittels Epsilon-Greedy-Ansatz
                if np.random.rand() < self.epsilon:
                    action = self.env.action_space.sample()  # Explore
                else:
                    action = np.argmax(self.q_table[state])  # Exploit

                # Buy / Sell Entscheidung
                next_observation, reward, terminated, truncated, info = self.env.step(
                    action
                )
                next_state = next_observation

                # Prüfung ob nächster State-Tuple bereits im Q-Table vorhanden ist.
                if next_state not in self.q_table:
                    self.q_table[next_state] = np.zeros(self.n_actions)

                # Q-Table aktualisieren -> Mittels Bellman-Equation aus dem Vorlesungsskript
                best_future_q = np.max(self.q_table[next_state])
                self.q_table[state][action] = (1 - self.learning_rate) * self.q_table[
                    state
                ][action] + self.learning_rate * (
                    reward + self.discount_factor * best_future_q
                )

                total_reward += reward
                state = next_state
                done = terminated or truncated

            # Epsilon verringern
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Alle 100 Schritte in der Konsole ausgeben
            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch}, Total Reward: {total_reward}, Epsilon: {self.epsilon:.4f}"
                )

        # Am Ende den Q-Table sowie den theoretisch maximal erreichbaren Profit ausgeben
        print(f"Maximum Profit:  {self.env.unwrapped.max_possible_profit()}")
        print("Trained Q-table:")
        for state, q_values in self.q_table.items():
            print(f"State: {state}, Q-values: {q_values}")

        plt.cla()
        self.env.unwrapped.render_all()
        plt.show()

#Trainieren des Agenten mit den Standardwerten und der Google-Aktie.
Agent(frame_bound=(5, 50)).train_agent()

#Trainieren mit eigener Aktienchart
#Agent(dataframe=pd.read_csv("VW.csv"), frame_bound=(5, 50)).train_agent()

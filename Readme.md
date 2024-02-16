# Codedokumentation zu RL_forTrading.py Projekt

## Preliminaries:

Es gibt grundsätzlich zwei Möglichkeiten, wie der Code die gewünschte Ausgabe liefert. Entweder verfügt man über die 
passende Hardware, nämlich eine Nvidia-Grafikkarte der neueren Generation, die CUDA unterstützt, 
oder man führt die Modellkalibrierung auf der CPU durch. Man sollte hier jedoch beachten, dass Letzteres sehr 
zeitaufwändig sein kann. Die Geschwindigkeit hängt von der Hardware ab. Auf meinem Laptop konnte ich nicht 
einmal eine Episode abschließen. Im folgenden wird kurz beschrieben, wie der Installationsprozess anläuft.

1. Zuerst wird eine passende Python-Distribution benötigt. Ich habe Python3.9 verwendet. CUDA wird aber bis 3.11 unterstützt.
	- Python Version: **Python 3.9.11** (https://www.python.org/downloads/)

2. Anschließend muss man sich entscheiden, ob man CUDA-Unterstützung (GPU) oder CPU-Unterstützung verwenden möchten.
 
- **CUDA Variante**: Details siehe: https://pytorch.org/get-started/locally/
	- CUDA 11.7 von https://developer.nvidia.com/cuda-zone installieren.
	- Verwenden des Befehl **pip install torch==1.13.1+cu117** zur Installation von PyTorch.
	- würde Installation via **pip** empfehlen
	- CUDA-Verfügbarkeit mit dem Befehl: **python -c "import torch; print(torch.cuda.is_available())"** überprüfen

**ACHTUNG!**: Die CPU-Variante wurde nicht getestet, sollte jedoch unproblematisch sein.
- **CPU Variante** (würde ich nicht empfehlen): Details siehe: https://pytorch.org/get-started/locally/
	- Verwenden des Befehls **pip3 install torch torchvision torchaudio** zur Installation von PyTorch.
	- Installation überprüfen: **import torch**

Verwendete packages:
- **Required packages**
	- numpy>=1.24.2
	- pandas>=1.5.3
	- matplotlib>=3.1.2
	- empyrical>=0.5.0
	- yahoofinancials>=1.5
	- Bei GPU-Unterstützung: **torch==1.13.1+cu117**, andernfalls **torch**
	- PyYAML>=6.0
	
Die erforderlichen Pakete können einfach über die Batch-Datei *get_packages_GPU.bat* oder *get_packages_CPU.bat* installiert werden. Die .txt-Dateien enthalten die entsprechenden Paketlisten.

## Projektstruktur:

Der Code ist als Visual Studio Projekt konzipiert, wobei *RL_forTrading.py* die Hauptdatei ist. Es gibt auch einen **RUN_project**-Ordner, der eine Batch-Datei enthält, 
um das Projekt zu starten. Die Abhängigkeiten im Projekt sind so definiert, dass ausschließlich relative Pfadangaben verwendet werden. Das Projekt sollte also reibungslos 
funktionieren. Im Verlauf des Projekts werden automatisch Unterordner erstellt. Die Projektstruktur lautet wie folgt:

	| -> config\config.yml [ein paar allgemeine Dinge wie zB Train und Test Dates, sowie die Hyperparameter]
	| -> Input: LaTeX, csv und einige json Files 	
	| -> Output: Ergebnispfad
	| -> Cpp_implementation: (unnötig) ex-ante volatilität in C++, pybind11 anbindung existiert in separatem Projekt
	| -> RUN_project: startet das Projekt [enthält ein batch Files]

	├───> RL_forTrading.py (Main File)
	├───config
	├───Cpp_implementation (irrelevant)
	│   ├───.vscode
	│   └───test
	├───GetPackages 
	├───Input
	│   ├───jsonFiles
	│   └───LaTeX
	├───outputs
	│   └───20230907-133040
	│       ├───models_DDQN_Bonds2017
	│       ├───models_DDQN_Bonds2018
	│       ├───models_DDQN_Bonds2019
	│       ├───models_DDQN_Bonds2020
	|
	├───RUN_project
	└───utils
		├───LaTeX
		│   └───__pycache__
		├───Logging
		│   └───__pycache__
		├───MACD
		│   └───__pycache__
		├───MoveFiles
		├───Neural_Networks
		│   └───__pycache__
		├───Plotting
		│   └───__pycache__
		├───testing
		│   └───__pycache__
		├───trading_env
		│   └───__pycache__
		└───__pycache__

## Aufbau:


Die Hauptdatei ist RL_forTrading.py. Diese kann auf verschiedene Arten gestartet werden: über die Befehlszeile, da sie 
keine zusätzlichen Benutzereingaben erfordert (also *python RL_forTrading.py* in der command-line), 
oder über Visual Studio bzw jede andere IDE. Innerhalb des if **__name__ == "__main__"**-Blocks können verschiedene Parameter festgelegt werden.

Erstens können die Asset-Klassen in einer Liste festgelegt werden. 
- **FX** (6 Märkte)
- **Equities** (6 Märkte)
- **Bonds** (12 Märkte)
- **Softs and Crops** (14 Märkte)
- **Metals** (8 Märkte)
- **Energies** (6 Märkte)


Zweitens kann der Benutzer ein bestimmtes Modell über den Parameter *models* angeben. Der String muss dem entsprechenden Ordner im "\outputs\" 
Verzeichnis entsprechen. Falls das Modell nicht gefunden wird, wird ein Fehler an entsprechender Stelle abgefangen und der Durchlauf beendet.

Training und Evaluation werden in separaten Durchläufen durchgeführt. Das bedeutet, dass nachdem der Code einmal durchgelaufen ist und die neuronalen Netze trainiert wurden, 
werden diese gespeichert. Beim nächsten Durchlauf kann der Name des Modells geändert werden, z. B. in 
- **'20230907-133040'**

Dieser Ordner enthält bereits alle Modelle 
sowie die dazugehörigen Ergebnisse. Wenn dieser Ordner angegeben wird, werden die Plots repliziert, die sich auch im PDF-Bericht befinden. Um den
Prozess zu vereinfachen, habe ich aber ein zweites Batch file erstellt, welches den Code über die Command-line aufruft. Das File heißt dann: 
RunProject_evaluate.bat

### Training
*Configure.py* konfiguriert eigentlich alles. Sprich die *Config-Klasse* erstellt Instanzen von *TradingEnv* und *DQN* (der Name ist ein wenig irreführend, denn er beinhaltet sowohl
DQN als auch DDQN). Der Training-Prozess findet dann im File *training.py* statt. Anschließend wird das Modell gesichert. Die Unterscheidung multitrain ist eigentlich nur dazu da,
ob das Training rein an einem Markt durchgeführt wird oder ob alle dafür herangezogen werden.

### Testing

Gleich wie oben. Nur jetzt wird das Modell für die Asset Klasse geladen, die MACD Strategie auf die Close-Preise ausgeführt und die Ergebnisse ausgewertet. Für die RL-Strategien, sowie
die Buy and Hold-Strategie geschieht das gleiche. Anschließend werden die LaTeX-Tabellen bzw. die Plots erstellt.

## Daten:

Die Daten enthalten keine vertraulichen Informationen, sollten jedoch ausschließlich für den Zweck dieses speziellen Projekts verwendet werden. 
Da die Daten sehr umfangreich sind, ist in der Regel keine Weitergabe gestattet. Aus gutem Grund gehen die Daten auch nicht über das Jahr 2020 hinaus.


## Dauer:

Auf meinem Rechner hat der Code ein bisschen mehr als eine Stunde gedauert. Allerdings gehört die Hardwarekonfiguration zu den bestehen, die man im Konsumersegment vor einem halben
Jahr kaufen konnte. Es ist also recht wahrscheinlich, dass der Code wesentlich länger braucht. 

Der Benutzer muss sich im Klaren sein, dass in Summe nicht ein großes Modell, sondern 24 Modell für einen Algorithmus erstellt werden (6 Märkte x 4 Testperioden). In Summe werden
also 48 modelle kalibriert.

